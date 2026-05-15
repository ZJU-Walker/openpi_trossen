"""Offline evaluation of a trained pi0/pi0.5 policy against a LeRobot dataset episode.

Loads a checkpoint, iterates frames of one episode, runs policy inference with
chunked re-inference (mirroring the deployed-client pattern), and writes:

  - actions.csv          : per-frame ground-truth vs. predicted action
  - joint_NN.png         : one figure per joint dim (gt solid, pred dashed)
  - joints_all.png       : 7x2 grid of all joints
  - stdout               : per-joint and overall MSE

Usage (LoRA config — default):
    cd /iris/u/kewalk/openpi_trossen
    export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data
    uv run examples/trossen_ai/eval_offline.py \
        --checkpoint-dir checkpoints/pi05_trossen_pack_with_human/pi05_pack_with_human_0514/29999

Usage (full finetune config):
    uv run examples/trossen_ai/eval_offline.py \
        --config-name pi05_trossen_pack_with_human_full \
        --checkpoint-dir checkpoints/pi05_trossen_pack_with_human_full/pi05_pack_with_human_full_0514/29999
"""

import csv
import dataclasses
import logging
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tyro

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def _strip_actions_from_repack(group: _transforms.Group) -> _transforms.Group:
    """Return a Group whose inputs are RepackTransforms with the 'actions' leaf removed.

    The training repack maps {"actions": "action", ...} but at inference we don't have
    ground-truth actions in the observation dict. Drop that leaf so the repack only
    looks up keys we actually provide.
    """
    new_inputs = []
    for t in group.inputs:
        if isinstance(t, _transforms.RepackTransform) and isinstance(t.structure, dict):
            stripped = {k: v for k, v in t.structure.items() if k != "actions"}
            new_inputs.append(_transforms.RepackTransform(stripped))
        else:
            new_inputs.append(t)
    return _transforms.Group(inputs=tuple(new_inputs), outputs=group.outputs)


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    """Path to a checkpoint step directory (e.g. checkpoints/.../29999)."""

    config_name: str = "pi05_trossen_pack_with_human"
    """Train config name registered in src/openpi/training/config.py."""

    episode_index: int = 0
    """Which episode to evaluate."""

    num_frames: int = -1
    """How many frames to evaluate from the start of the episode. -1 = whole episode."""

    chunk_size: int = 50
    """How many actions to consume per inference before re-inferring.
    Must be <= model action_horizon (50). chunk_size=1 means re-infer every frame."""

    output_dir: str | None = None
    """Where to write outputs. Default: eval_outputs/<config>_<exp>_<step>/episode_<N>/."""

    prompt: str | None = None
    """Prompt override. Default: pulled from the train config's data.default_prompt."""

    pytorch_device: str = "cuda"
    """PyTorch device. Ignored for JAX checkpoints."""


def to_chw_uint8(img) -> np.ndarray:
    """LeRobot videos may come in as torch CHW float [0,1] OR HWC uint8 depending on backend.
    The downstream aloha_policy._decode_aloha expects CHW (it rearranges c h w -> h w c).
    Normalize to CHW uint8."""
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    # Float in [0,1] -> uint8
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    # Detect layout: CHW has 3 in axis 0 (and h,w much bigger); HWC has 3 in axis -1.
    if img.ndim != 3:
        raise ValueError(f"Image must be 3D, got shape {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        return img  # already CHW
    if img.shape[-1] == 3:
        return np.transpose(img, (2, 0, 1))  # HWC -> CHW
    raise ValueError(f"Cannot infer image layout from shape {img.shape}")


def _resolve_output_dir(args: Args, ckpt_path: pathlib.Path) -> pathlib.Path:
    if args.output_dir:
        return pathlib.Path(args.output_dir)
    # ckpt_path = .../<config>/<exp>/<step>
    step = ckpt_path.name
    exp = ckpt_path.parent.name
    cfg = ckpt_path.parent.parent.name
    return pathlib.Path("eval_outputs") / f"{cfg}_{exp}_{step}" / f"episode_{args.episode_index}"


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if "HF_LEROBOT_HOME" not in os.environ:
        logger.warning(
            "HF_LEROBOT_HOME is not set. LeRobot will try to fetch the dataset from the Hub. "
            "For local Trossen data, run: export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data"
        )

    ckpt_path = pathlib.Path(args.checkpoint_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {ckpt_path}")

    output_dir = _resolve_output_dir(args, ckpt_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    train_config = _config.get_config(args.config_name)

    # data_config gives us repack_transforms (dataset key -> policy key) and repo_id.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError(f"Config {args.config_name!r} has no repo_id.")
    prompt = (
        args.prompt
        or getattr(data_config, "default_prompt", None)
        or getattr(train_config.data, "default_prompt", None)
        or (train_config.policy_metadata or {}).get("default_prompt")
    )
    if prompt is None:
        logger.warning("No prompt provided and config has no default_prompt; using empty string.")
        prompt = ""
    logger.info(f"Using prompt: {prompt!r}")

    action_horizon = train_config.model.action_horizon
    if args.chunk_size < 1 or args.chunk_size > action_horizon:
        raise ValueError(f"--chunk-size must be in [1, {action_horizon}], got {args.chunk_size}")

    logger.info(f"Loading policy from {ckpt_path}")
    # The training repack_transform expects an "actions" leaf (mapped from "action"), but at
    # inference we don't have ground-truth actions. Build an inference-only variant of the repack
    # that drops the "actions" entry — same as how the websocket server adapter operates.
    inference_repack = _strip_actions_from_repack(data_config.repack_transforms)
    policy = _policy_config.create_trained_policy(
        train_config,
        ckpt_path,
        repack_transforms=inference_repack,
        default_prompt=prompt,
        pytorch_device=args.pytorch_device,
    )

    # Import LeRobot only after we've validated everything else (slow import).
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    if args.episode_index >= meta.total_episodes:
        raise ValueError(
            f"episode_index={args.episode_index} >= total_episodes={meta.total_episodes}"
        )

    # Dataset without delta_timestamps -> each frame's "action" is a single (action_dim,) vector.
    dataset = LeRobotDataset(repo_id)
    ep_from, ep_to = _episode_bounds(dataset, meta, args.episode_index)
    frame_indices = list(range(ep_from, ep_to))
    if args.num_frames > 0:
        frame_indices = frame_indices[: args.num_frames]
    T = len(frame_indices)
    logger.info(f"Evaluating episode {args.episode_index}: frames [{ep_from}, {ep_to}) -> using T={T}")

    # Infer action_dim from the dataset itself. The model's internal action_dim (32 for pi0.5)
    # is padded; aloha_policy.AlohaOutputs slices back to 14 for Trossen/Aloha. We trust the
    # dataset feature shape as the source of truth.
    sample = dataset[frame_indices[0]]
    sample_action = sample["action"]
    if hasattr(sample_action, "numpy"):
        sample_action = sample_action.numpy()
    action_dim = int(np.asarray(sample_action).shape[-1])
    logger.info(f"Dataset action_dim={action_dim} (model.action_dim={train_config.model.action_dim})")

    gt_actions = np.zeros((T, action_dim), dtype=np.float32)
    pred_actions = np.zeros((T, action_dim), dtype=np.float32)

    action_chunk: np.ndarray | None = None
    chunk_idx = args.chunk_size  # forces inference on first step
    infer_times_ms: list[float] = []

    image_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )

    for t, fidx in enumerate(frame_indices):
        frame = dataset[fidx]
        gt = frame["action"]
        if hasattr(gt, "numpy"):
            gt = gt.numpy()
        gt_actions[t] = np.asarray(gt, dtype=np.float32)

        if chunk_idx >= args.chunk_size:
            state = frame["observation.state"]
            if hasattr(state, "numpy"):
                state = state.numpy()
            obs = {
                "observation.state": np.asarray(state, dtype=np.float32),
                "prompt": prompt,
            }
            for k in image_keys:
                obs[k] = to_chw_uint8(frame[k])
            t0 = time.monotonic()
            result = policy.infer(obs)
            infer_times_ms.append((time.monotonic() - t0) * 1000.0)
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[0] != action_horizon:
                logger.warning(
                    f"Unexpected action_chunk shape {action_chunk.shape}; "
                    f"expected ({action_horizon}, *)"
                )
            if action_chunk.shape[-1] != action_dim:
                # Slice or pad to the dataset's action_dim — first action_dim cols are the real joints.
                action_chunk = action_chunk[:, :action_dim]
            chunk_idx = 0

        pred_actions[t] = action_chunk[chunk_idx]
        chunk_idx += 1

        if (t + 1) % 50 == 0 or t == T - 1:
            logger.info(f"  frame {t+1}/{T}")

    # --- Save CSV ---
    csv_path = output_dir / "actions.csv"
    headers = (
        ["frame_index"]
        + [f"gt_{i}" for i in range(action_dim)]
        + [f"pred_{i}" for i in range(action_dim)]
    )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for t, fidx in enumerate(frame_indices):
            w.writerow([fidx, *gt_actions[t].tolist(), *pred_actions[t].tolist()])
    logger.info(f"Wrote CSV: {csv_path}")

    # --- MSE ---
    sq_err = (pred_actions - gt_actions) ** 2
    mse_per_joint = sq_err.mean(axis=0)
    overall_mse = sq_err.mean()
    print("\n=== MSE per joint ===")
    for j in range(action_dim):
        print(f"  joint_{j:02d}: {mse_per_joint[j]:.6f}")
    print(f"\nOverall MSE: {overall_mse:.6f}")
    if infer_times_ms:
        print(f"Inferences: {len(infer_times_ms)}  mean: {np.mean(infer_times_ms):.1f} ms  "
              f"median: {np.median(infer_times_ms):.1f} ms")

    # --- Per-joint plots ---
    xs = np.arange(T)
    for j in range(action_dim):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, gt_actions[:, j], label="ground truth", linewidth=1.5)
        ax.plot(xs, pred_actions[:, j], label="predicted", linewidth=1.0, linestyle="--")
        ax.set_xlabel("timestep")
        ax.set_ylabel(f"action[{j}]")
        ax.set_title(f"joint {j:02d}   MSE={mse_per_joint[j]:.5f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"joint_{j:02d}.png", dpi=110)
        plt.close(fig)

    # --- Summary grid ---
    rows, cols = 7, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 18), sharex=True)
    for j in range(action_dim):
        ax = axes[j // cols, j % cols]
        ax.plot(xs, gt_actions[:, j], label="gt", linewidth=1.0)
        ax.plot(xs, pred_actions[:, j], label="pred", linewidth=0.8, linestyle="--")
        ax.set_title(f"joint {j:02d}  mse={mse_per_joint[j]:.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(action_dim, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(
        f"{args.config_name}  ep{args.episode_index}  T={T}  chunk_size={args.chunk_size}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / "joints_all.png", dpi=110)
    plt.close(fig)
    logger.info(f"Wrote {action_dim} per-joint plots + joints_all.png to {output_dir}")


def _episode_bounds(dataset, meta, episode_index: int) -> tuple[int, int]:
    """Return [from, to) frame indices for `episode_index`.

    Tries multiple LeRobot layouts in order:
      1. v0.1.0+: dataset.episode_data_index["from" / "to"] (tensor per episode)
      2. Older "from_index"/"to_index" keys on meta.episodes entries
      3. Last resort: scan frames via dataset[i]["episode_index"] (slow).
    """
    edi = getattr(dataset, "episode_data_index", None)
    if isinstance(edi, dict) and "from" in edi and "to" in edi:
        f, t = edi["from"][episode_index], edi["to"][episode_index]
        return int(f.item() if hasattr(f, "item") else f), int(t.item() if hasattr(t, "item") else t)

    if hasattr(meta, "episodes"):
        ep = meta.episodes[episode_index]
        if isinstance(ep, dict) and "from_index" in ep and "to_index" in ep:
            return int(ep["from_index"]), int(ep["to_index"])

    # Slow fallback
    logger.info("Falling back to scanning frames for episode boundaries.")
    from_idx, to_idx = None, None
    for i in range(len(dataset)):
        ei = dataset[i]["episode_index"]
        if hasattr(ei, "item"):
            ei = ei.item()
        if ei == episode_index and from_idx is None:
            from_idx = i
        elif from_idx is not None and ei != episode_index:
            to_idx = i
            break
    if from_idx is None:
        raise ValueError(f"Episode {episode_index} not found in dataset")
    if to_idx is None:
        to_idx = len(dataset)
    return from_idx, to_idx


if __name__ == "__main__":
    main(tyro.cli(Args))
