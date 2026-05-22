#!/usr/bin/env python3
"""Continuity diagnostics for `examples/trossen_ai/eval_offline.py` outputs."""

import argparse
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _resolve_actions_csv(path: pathlib.Path) -> pathlib.Path:
    if path.is_file():
        return path
    actions_csv = path / "actions.csv"
    if actions_csv.exists():
        return actions_csv
    matches = sorted(path.glob("episode_*/actions.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No actions.csv found under {path}")
    raise ValueError(f"Multiple actions.csv files found under {path}; pass one episode directory explicitly.")


def _load_actions(actions_csv: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with actions_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]

    data = np.asarray(rows, dtype=np.float64)
    gt_cols = [i for i, name in enumerate(header) if name.startswith("gt_")]
    pred_cols = [i for i, name in enumerate(header) if name.startswith("pred_")]
    if not gt_cols or len(gt_cols) != len(pred_cols):
        raise ValueError(f"Could not infer matching gt_/pred_ columns from {actions_csv}")
    return data[:, 0].astype(np.int64), data[:, gt_cols], data[:, pred_cols]


def _summarize_steps(name: str, steps: np.ndarray) -> dict[str, float]:
    if steps.size == 0:
        return {
            f"{name}_mean_l2": 0.0,
            f"{name}_p95_l2": 0.0,
            f"{name}_max_l2": 0.0,
            f"{name}_mean_max_abs": 0.0,
            f"{name}_max_abs": 0.0,
        }
    l2 = np.linalg.norm(steps, axis=-1)
    max_abs = np.max(np.abs(steps), axis=-1)
    return {
        f"{name}_mean_l2": float(np.mean(l2)),
        f"{name}_p95_l2": float(np.percentile(l2, 95)),
        f"{name}_max_l2": float(np.max(l2)),
        f"{name}_mean_max_abs": float(np.mean(max_abs)),
        f"{name}_max_abs": float(np.max(max_abs)),
    }


def _format_summary(summary: dict[str, float | int | list[int]]) -> str:
    lines = [
        "=== Continuity summary ===",
        f"frames: {summary['frames']}",
        f"action_dim: {summary['action_dim']}",
        f"chunk_size: {summary['chunk_size']}",
        f"boundary_count: {summary['boundary_count']}",
        "",
        "Predicted action deltas:",
        f"  all mean L2:        {summary['pred_all_mean_l2']:.6f}",
        f"  all p95 L2:         {summary['pred_all_p95_l2']:.6f}",
        f"  all max L2:         {summary['pred_all_max_l2']:.6f}",
        f"  non-boundary mean:  {summary['pred_non_boundary_mean_l2']:.6f}",
        f"  boundary mean:      {summary['pred_boundary_mean_l2']:.6f}",
        f"  boundary max:       {summary['pred_boundary_max_l2']:.6f}",
        f"  boundary/non ratio: {summary['pred_boundary_to_non_boundary_l2_ratio']:.3f}",
        "",
        "Ground-truth action deltas:",
        f"  all mean L2:        {summary['gt_all_mean_l2']:.6f}",
        f"  boundary mean:      {summary['gt_boundary_mean_l2']:.6f}",
        "",
        "Predicted higher-order smoothness:",
        f"  acceleration mean L2: {summary['pred_accel_mean_l2']:.6f}",
        f"  acceleration max L2:  {summary['pred_accel_max_l2']:.6f}",
        f"  jerk mean L2:         {summary['pred_jerk_mean_l2']:.6f}",
        f"  jerk max L2:          {summary['pred_jerk_max_l2']:.6f}",
        "",
        f"worst boundary frame indices: {summary['worst_boundary_frames']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate action continuity from eval_offline actions.csv.")
    parser.add_argument("path", help="Path to actions.csv, an episode dir, or an eval output dir.")
    parser.add_argument("--chunk-size", type=int, default=25, help="Chunk size used by eval_offline.")
    parser.add_argument("--output-dir", default=None, help="Directory for continuity outputs. Default: beside CSV.")
    args = parser.parse_args()

    actions_csv = _resolve_actions_csv(pathlib.Path(args.path).resolve())
    output_dir = pathlib.Path(args.output_dir).resolve() if args.output_dir else actions_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices, gt_actions, pred_actions = _load_actions(actions_csv)
    num_frames, action_dim = pred_actions.shape
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    gt_steps = np.diff(gt_actions, axis=0)
    pred_steps = np.diff(pred_actions, axis=0)
    step_frames = frame_indices[1:]
    boundary_frames = np.arange(args.chunk_size, num_frames, args.chunk_size)
    boundary_step_mask = np.zeros(num_frames - 1, dtype=bool)
    boundary_step_mask[boundary_frames - 1] = True

    pred_boundary_steps = pred_steps[boundary_step_mask]
    pred_non_boundary_steps = pred_steps[~boundary_step_mask]
    gt_boundary_steps = gt_steps[boundary_step_mask]

    pred_boundary_l2 = np.linalg.norm(pred_boundary_steps, axis=-1) if len(pred_boundary_steps) else np.array([])
    worst_order = np.argsort(pred_boundary_l2)[::-1][:5] if len(pred_boundary_l2) else np.array([], dtype=int)
    worst_boundary_frames = boundary_frames[worst_order].astype(int).tolist() if len(boundary_frames) else []

    summary: dict[str, float | int | list[int]] = {
        "frames": int(num_frames),
        "action_dim": int(action_dim),
        "chunk_size": int(args.chunk_size),
        "boundary_count": len(boundary_frames),
        "worst_boundary_frames": worst_boundary_frames,
    }
    summary.update(_summarize_steps("pred_all", pred_steps))
    summary.update(_summarize_steps("pred_boundary", pred_boundary_steps))
    summary.update(_summarize_steps("pred_non_boundary", pred_non_boundary_steps))
    summary.update(_summarize_steps("gt_all", gt_steps))
    summary.update(_summarize_steps("gt_boundary", gt_boundary_steps))
    pred_accel = np.diff(pred_steps, axis=0)
    pred_jerk = np.diff(pred_accel, axis=0)
    summary.update(_summarize_steps("pred_accel", pred_accel))
    summary.update(_summarize_steps("pred_jerk", pred_jerk))
    non_boundary_mean = float(summary["pred_non_boundary_mean_l2"])
    summary["pred_boundary_to_non_boundary_l2_ratio"] = (
        float(summary["pred_boundary_mean_l2"]) / non_boundary_mean if non_boundary_mean > 0 else float("inf")
    )

    summary_text = _format_summary(summary)
    print(summary_text)
    (output_dir / "continuity_summary.txt").write_text(summary_text + "\n")
    (output_dir / "continuity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    pred_l2 = np.linalg.norm(pred_steps, axis=-1)
    gt_l2 = np.linalg.norm(gt_steps, axis=-1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(step_frames, pred_l2, label="pred |delta action|", linewidth=1.2)
    ax.plot(step_frames, gt_l2, label="gt |delta action|", linewidth=1.0, alpha=0.7)
    for frame in boundary_frames:
        ax.axvline(frame_indices[frame], color="tab:red", alpha=0.18, linewidth=0.8)
    ax.set_xlabel("frame index")
    ax.set_ylabel("L2 step delta")
    ax.set_title("Action step deltas; red lines are chunk boundaries")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "continuity_step_deltas.png", dpi=120)
    plt.close(fig)

    if len(pred_boundary_steps):
        boundary_abs = np.abs(pred_boundary_steps)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(action_dim), boundary_abs.max(axis=0), label="max boundary jump")
        ax.bar(np.arange(action_dim), np.abs(pred_non_boundary_steps).max(axis=0), alpha=0.5, label="max non-boundary")
        ax.set_xlabel("action dimension")
        ax.set_ylabel("max abs delta")
        ax.set_title("Per-joint jump size")
        ax.grid(visible=True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "continuity_joint_jumps.png", dpi=120)
        plt.close(fig)

    print(f"\nWrote continuity outputs to: {output_dir}")


if __name__ == "__main__":
    main()
