"""Offline HIERARCHICAL evaluation: per-frame subtask prompts driving a pi0.5 policy.

Extends eval_offline.py: instead of one fixed prompt, the language prompt at
each frame comes from either the high-level Qwen predictions (stage-1 JSON from
Qwen3-VL/qwen-vl-finetune/tools/predict_subtasks_lerobot.py) or the manual
ground-truth segment CSV (oracle mode, isolating low-level error). The policy
re-infers when its action chunk is exhausted OR when the prompt switches
(matching how the deployed hierarchical controller behaves).

Writes per episode:
  actions.csv          frame, subtask, prompt, gt_* and pred_* per joint
  summary.json         per-joint MAE/MSE, per-subtask breakdown, infer stats
  joint_NN.png         per-joint gt vs pred, subtask shading, re-infer markers
  joints_all.png       7x2 grid
  error_per_joint.png  MAE bar chart
  error_over_time.png  mean |err| across joints vs time, subtask shading

Usage (oracle prompts from the manual CSV):
    cd /iris/u/kewalk/openpi_trossen
    export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data
    uv run examples/trossen_ai/eval_offline_hierarchical.py \
        --config-name pi05_trossen_block_mem_subtask_full_delta_rtc \
        --checkpoint-dir checkpoints/pi05_trossen_block_mem_subtask_full_delta_rtc/<exp>/7000 \
        --episode-index 27 --prompt-source gt

Usage (Qwen-predicted prompts, after running stage 1):
    uv run examples/trossen_ai/eval_offline_hierarchical.py \
        ... --episode-index 27 --prompt-source qwen \
        --subtask-json <stage1_output>/subtask_preds_ep000027.json
"""

import csv
import dataclasses
import json
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

# subtask slug -> low-level prompt (matches the training config's
# PromptFromSubtaskSegments mapping; override with --prompt-map-json).
DEFAULT_PROMPT_MAP = {
    "observe_human": "wait and observe the human",
    "put_green_block_to_plate": "put the green block to the plate",
    "put_yellow_block_to_plate": "put the yellow block to the plate",
}

SUBTASK_COLORS = {
    "observe_human": "#bdbdbd",
    "put_green_block_to_plate": "#74c476",
    "put_yellow_block_to_plate": "#fdd835",
}


def _strip_actions_from_repack(group: _transforms.Group) -> _transforms.Group:
    new_inputs = []
    for t in group.inputs:
        if isinstance(t, _transforms.RepackTransform) and isinstance(t.structure, dict):
            stripped = {k: v for k, v in t.structure.items()
                        if k not in ("actions", "episode_index", "frame_index")}
            new_inputs.append(_transforms.RepackTransform(stripped))
        else:
            new_inputs.append(t)
    return _transforms.Group(inputs=tuple(new_inputs), outputs=group.outputs)


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    """Path to a checkpoint step directory (e.g. checkpoints/.../7000)."""

    config_name: str = "pi05_trossen_block_mem_subtask_full_delta_rtc"
    """Train config name registered in src/openpi/training/config.py."""

    episode_index: int = 27
    """Which episode to evaluate (default: first val episode)."""

    prompt_source: str = "gt"
    """Where per-frame prompts come from: 'qwen' (stage-1 JSON), 'gt' (oracle
    from --labels-csv), or 'fixed' (single --prompt, eval_offline.py behavior)."""

    subtask_json: str | None = None
    """Stage-1 predictions JSON for this episode (required for prompt_source=qwen)."""

    labels_csv: str = "/iris/projects/humanoid/trossen_data/labels/subtask_segments_0528_merge_manual.csv"
    """Manual segment CSV (required for prompt_source=gt; also used for shading)."""

    prompt_map_json: str | None = None
    """Optional JSON file overriding the subtask slug -> low-level prompt map."""

    prompt: str | None = None
    """Fixed prompt for prompt_source=fixed."""

    num_frames: int = -1
    """How many frames to evaluate from the start of the episode. -1 = all."""

    chunk_size: int = 50
    """Actions consumed per inference before re-inferring (<= action_horizon)."""

    output_dir: str | None = None
    """Default: eval_outputs/<config>_<exp>_<step>/episode_<N>_<prompt_source>/."""

    pytorch_device: str = "cuda"


def to_chw_uint8(img) -> np.ndarray:
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim != 3:
        raise ValueError(f"Image must be 3D, got shape {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        return img
    if img.shape[-1] == 3:
        return np.transpose(img, (2, 0, 1))
    raise ValueError(f"Cannot infer image layout from shape {img.shape}")


def _load_gt_subtasks(labels_csv: str, dataset_name: str, episode_index: int) -> dict[int, str]:
    table: dict[int, str] = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != dataset_name or int(row["episode_id"]) != episode_index:
                continue
            for frame in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
                table[frame] = row["subtask"]
    return table


def _load_qwen_subtasks(subtask_json: str, episode_index: int, T: int) -> dict[int, str]:
    data = json.loads(pathlib.Path(subtask_json).read_text())
    if int(data["episode"]) != episode_index:
        raise ValueError(
            f"--subtask-json is for episode {data['episode']}, not {episode_index}")
    # Forward-fill over stride gaps so every frame has the latest prediction.
    table: dict[int, str] = {}
    last = None
    preds = {int(r["frame"]): r["pred"] for r in data["frames"]}
    for t in range(T):
        last = preds.get(t, last)
        if last is not None:
            table[t] = last
    return table


def _subtask_segments(subtasks: dict[int, str], T: int) -> list[tuple[int, int, str]]:
    """Contiguous (start, end, slug) runs of the active subtask over [0, T)."""
    segments = []
    start, cur = 0, subtasks.get(0)
    for t in range(1, T):
        s = subtasks.get(t, cur)
        if s != cur:
            segments.append((start, t - 1, cur))
            start, cur = t, s
    segments.append((start, T - 1, cur))
    return [(a, b, s) for a, b, s in segments if s is not None]


def _shade_subtasks(ax, segments, alpha=0.18):
    for start, end, slug in segments:
        ax.axvspan(start, end + 1, color=SUBTASK_COLORS.get(slug, "#90caf9"), alpha=alpha)


def _resolve_output_dir(args: Args, ckpt_path: pathlib.Path) -> pathlib.Path:
    if args.output_dir:
        return pathlib.Path(args.output_dir)
    step = ckpt_path.name
    exp = ckpt_path.parent.name
    cfg = ckpt_path.parent.parent.name
    return (pathlib.Path("eval_outputs") / f"{cfg}_{exp}_{step}"
            / f"episode_{args.episode_index}_{args.prompt_source}")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if "HF_LEROBOT_HOME" not in os.environ:
        logger.warning("HF_LEROBOT_HOME is not set; LeRobot may hit the Hub. "
                       "Run: export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data")
    if args.prompt_source not in ("qwen", "gt", "fixed"):
        raise ValueError("--prompt-source must be qwen, gt, or fixed")
    if args.prompt_source == "qwen" and not args.subtask_json:
        raise ValueError("--subtask-json is required for --prompt-source qwen")

    ckpt_path = pathlib.Path(args.checkpoint_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {ckpt_path}")
    output_dir = _resolve_output_dir(args, ckpt_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    prompt_map = dict(DEFAULT_PROMPT_MAP)
    if args.prompt_map_json:
        prompt_map.update(json.loads(pathlib.Path(args.prompt_map_json).read_text()))

    train_config = _config.get_config(args.config_name)
    # The subtask config injects training prompts via PromptFromSubtaskSegments,
    # which requires episode_index/frame_index and would override obs["prompt"].
    # At eval the prompt comes from the high-level policy (or the oracle CSV),
    # so drop that transform and drive prompts through the observation dict.
    if getattr(train_config.data, "prompt_input_transform", None) is not None:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, prompt_input_transform=None),
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError(f"Config {args.config_name!r} has no repo_id.")

    action_horizon = train_config.model.action_horizon
    if args.chunk_size < 1 or args.chunk_size > action_horizon:
        raise ValueError(f"--chunk-size must be in [1, {action_horizon}]")

    logger.info(f"Loading policy from {ckpt_path}")
    inference_repack = _strip_actions_from_repack(data_config.repack_transforms)
    policy = _policy_config.create_trained_policy(
        train_config,
        ckpt_path,
        repack_transforms=inference_repack,
        default_prompt=args.prompt or "",
        pytorch_device=args.pytorch_device,
    )

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    if args.episode_index >= meta.total_episodes:
        raise ValueError(f"episode_index={args.episode_index} >= {meta.total_episodes}")
    dataset = LeRobotDataset(repo_id)
    ep_from, ep_to = _episode_bounds(dataset, meta, args.episode_index)
    frame_indices = list(range(ep_from, ep_to))
    if args.num_frames > 0:
        frame_indices = frame_indices[: args.num_frames]
    T = len(frame_indices)
    logger.info(f"Episode {args.episode_index}: frames [{ep_from}, {ep_to}) -> T={T}")

    # --- Per-frame subtask + prompt schedule (episode-local frame numbers) ---
    gt_subtasks = _load_gt_subtasks(args.labels_csv, repo_id, args.episode_index)
    if args.prompt_source == "qwen":
        subtasks = _load_qwen_subtasks(args.subtask_json, args.episode_index, T)
    elif args.prompt_source == "gt":
        subtasks = {t: gt_subtasks[t] for t in range(T) if t in gt_subtasks}
        if not subtasks:
            raise ValueError(f"No GT segments for episode {args.episode_index} in {args.labels_csv}")
    else:
        fixed = args.prompt or getattr(data_config, "default_prompt", "") or ""
        subtasks = {t: "fixed" for t in range(T)}
        prompt_map = {"fixed": fixed}

    def prompt_at(t: int) -> tuple[str, str]:
        slug = subtasks.get(t)
        if slug is None:  # before first prediction / unlabeled frame
            slug = next(iter(subtasks.values()))
        return slug, prompt_map.get(slug, prompt_map.get("observe_human", ""))

    sample = dataset[frame_indices[0]]
    sample_action = sample["action"]
    if hasattr(sample_action, "numpy"):
        sample_action = sample_action.numpy()
    action_dim = int(np.asarray(sample_action).shape[-1])

    gt_actions = np.zeros((T, action_dim), dtype=np.float32)
    pred_actions = np.zeros((T, action_dim), dtype=np.float32)
    active_slugs: list[str] = []

    action_chunk: np.ndarray | None = None
    chunk_idx = args.chunk_size
    active_prompt: str | None = None
    infer_times_ms: list[float] = []
    reinfer_points: list[dict] = []

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

        slug, prompt = prompt_at(t)
        active_slugs.append(slug)

        # Re-infer when the chunk is exhausted OR the subtask prompt changed.
        if chunk_idx >= args.chunk_size or prompt != active_prompt:
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
            action_chunk = np.asarray(result["actions"], dtype=np.float32)[:, :action_dim]
            reinfer_points.append({
                "t": t,
                "reason": "switch" if (active_prompt is not None and prompt != active_prompt
                                       and chunk_idx < args.chunk_size) else "chunk",
                "subtask": slug,
            })
            active_prompt = prompt
            chunk_idx = 0

        pred_actions[t] = action_chunk[chunk_idx]
        chunk_idx += 1
        if (t + 1) % 50 == 0 or t == T - 1:
            logger.info(f"  frame {t+1}/{T}  ({len(infer_times_ms)} inferences)")

    # --- CSV ---
    csv_path = output_dir / "actions.csv"
    headers = (["frame", "subtask", "prompt"]
               + [f"gt_{i}" for i in range(action_dim)]
               + [f"pred_{i}" for i in range(action_dim)])
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for t in range(T):
            slug, prompt = active_slugs[t], prompt_map.get(active_slugs[t], "")
            w.writerow([t, slug, prompt, *gt_actions[t].tolist(), *pred_actions[t].tolist()])
    logger.info(f"Wrote {csv_path}")

    # --- Metrics ---
    err = pred_actions - gt_actions
    mae_per_joint = np.abs(err).mean(axis=0)
    mse_per_joint = (err ** 2).mean(axis=0)
    per_subtask: dict[str, dict] = {}
    for slug in sorted(set(active_slugs)):
        mask = np.array([s == slug for s in active_slugs])
        per_subtask[slug] = {
            "frames": int(mask.sum()),
            "mae": float(np.abs(err[mask]).mean()),
            "mse": float((err[mask] ** 2).mean()),
            "mae_per_joint": np.abs(err[mask]).mean(axis=0).tolist(),
        }

    summary = {
        "config": args.config_name,
        "checkpoint": str(ckpt_path),
        "episode_index": args.episode_index,
        "prompt_source": args.prompt_source,
        "subtask_json": args.subtask_json,
        "chunk_size": args.chunk_size,
        "T": T,
        "overall_mae": float(np.abs(err).mean()),
        "overall_mse": float((err ** 2).mean()),
        "mae_per_joint": mae_per_joint.tolist(),
        "mse_per_joint": mse_per_joint.tolist(),
        "per_subtask": per_subtask,
        "num_inferences": len(infer_times_ms),
        "reinfer_points": reinfer_points,
        "infer_ms_mean": float(np.mean(infer_times_ms)) if infer_times_ms else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== {args.prompt_source} prompts | episode {args.episode_index} ===")
    print(f"overall MAE: {summary['overall_mae']:.5f}   MSE: {summary['overall_mse']:.6f}")
    for slug, m in per_subtask.items():
        print(f"  {slug:<28} frames={m['frames']:<5} MAE={m['mae']:.5f}")
    print(f"inferences: {len(infer_times_ms)} (switch-triggered: "
          f"{sum(r['reason'] == 'switch' for r in reinfer_points)})")

    # --- Figures ---
    xs = np.arange(T)
    segments = _subtask_segments({t: s for t, s in enumerate(active_slugs)}, T)
    switch_ts = [r["t"] for r in reinfer_points if r["reason"] == "switch"]

    for j in range(action_dim):
        fig, ax = plt.subplots(figsize=(10, 4))
        _shade_subtasks(ax, segments)
        ax.plot(xs, gt_actions[:, j], label="ground truth", linewidth=1.5)
        ax.plot(xs, pred_actions[:, j], label="predicted", linewidth=1.0, linestyle="--")
        for st in switch_ts:
            ax.axvline(st, color="red", linewidth=0.6, alpha=0.6)
        ax.set_xlabel("timestep")
        ax.set_ylabel(f"action[{j}]")
        ax.set_title(f"joint {j:02d}   MAE={mae_per_joint[j]:.5f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"joint_{j:02d}.png", dpi=110)
        plt.close(fig)

    rows, cols = 7, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 18), sharex=True)
    for j in range(action_dim):
        ax = axes[j // cols, j % cols]
        _shade_subtasks(ax, segments, alpha=0.15)
        ax.plot(xs, gt_actions[:, j], label="gt", linewidth=1.0)
        ax.plot(xs, pred_actions[:, j], label="pred", linewidth=0.8, linestyle="--")
        ax.set_title(f"joint {j:02d}  mae={mae_per_joint[j]:.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(action_dim, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(f"{args.config_name}  ep{args.episode_index}  prompts={args.prompt_source}  "
                 f"T={T}  chunk={args.chunk_size}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / "joints_all.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(action_dim), mae_per_joint, color="#1f77b4")
    ax.set_xlabel("joint")
    ax.set_ylabel("MAE")
    ax.set_title(f"per-joint MAE  (overall {summary['overall_mae']:.5f})")
    ax.set_xticks(np.arange(action_dim))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "error_per_joint.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    _shade_subtasks(ax, segments)
    ax.plot(xs, np.abs(err).mean(axis=1), linewidth=1.0, color="#d62728")
    for st in switch_ts:
        ax.axvline(st, color="red", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean |error| across joints")
    ax.set_title(f"action error over time  ep{args.episode_index}  prompts={args.prompt_source}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "error_over_time.png", dpi=110)
    plt.close(fig)
    logger.info(f"Wrote figures to {output_dir}")


def _episode_bounds(dataset, meta, episode_index: int) -> tuple[int, int]:
    edi = getattr(dataset, "episode_data_index", None)
    if isinstance(edi, dict) and "from" in edi and "to" in edi:
        f, t = edi["from"][episode_index], edi["to"][episode_index]
        return int(f.item() if hasattr(f, "item") else f), int(t.item() if hasattr(t, "item") else t)
    if hasattr(meta, "episodes"):
        ep = meta.episodes[episode_index]
        if isinstance(ep, dict) and "from_index" in ep and "to_index" in ep:
            return int(ep["from_index"]), int(ep["to_index"])
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
    return from_idx, to_idx if to_idx is not None else len(dataset)


if __name__ == "__main__":
    main(tyro.cli(Args))
