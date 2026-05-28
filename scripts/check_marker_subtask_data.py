#!/usr/bin/env python3
"""Validate marker handover subtask labels for pi0.5 training."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/iris/projects/humanoid/trossen_data")
DEFAULT_MERGED_REPO = "marker_handover_0526"
DEFAULT_LABELS = DEFAULT_DATA_ROOT / "scripts/labels/subtask_segments_0526_auto.csv"

SOURCE_EPISODE_OFFSETS = {
    "data_robot_give_0526": 0,
    "data_robot_pull_0526": 30,
}

SUBTASK_PROMPTS = {
    "keep_open": "keep the gripper open",
    "close": "close the gripper",
    "keep_closed": "keep the gripper closed",
    "open": "open the gripper",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_episode_lengths(dataset_root: Path) -> dict[int, int]:
    episodes = load_jsonl(dataset_root / "meta/episodes.jsonl")
    return {int(row["episode_index"]): int(row["length"]) for row in episodes}


def load_segments(labels_path: Path) -> tuple[dict[int, list[tuple[int, int, str]]], int]:
    segments: dict[int, list[tuple[int, int, str]]] = {}
    rows = 0

    with labels_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"dataset", "episode_id", "start_frame", "end_frame", "subtask"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Missing label columns: {sorted(missing_columns)}")

        for row in reader:
            rows += 1
            dataset = row["dataset"]
            subtask = row["subtask"]
            if dataset not in SOURCE_EPISODE_OFFSETS:
                raise ValueError(f"Unknown source dataset in labels: {dataset}")
            if subtask not in SUBTASK_PROMPTS:
                raise ValueError(f"Unknown subtask in labels: {subtask}")

            episode_index = SOURCE_EPISODE_OFFSETS[dataset] + int(row["episode_id"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            segments.setdefault(episode_index, []).append((start_frame, end_frame, subtask))

    for episode_segments in segments.values():
        episode_segments.sort()

    return segments, rows


def validate_coverage(
    episode_lengths: dict[int, int],
    segments: dict[int, list[tuple[int, int, str]]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    errors = []

    for episode_index, length in sorted(episode_lengths.items()):
        coverage: list[str | None] = [None] * length
        for start_frame, end_frame, subtask in segments.get(episode_index, []):
            if start_frame < 0 or end_frame < start_frame or end_frame >= length:
                errors.append(
                    f"episode {episode_index}: invalid segment {start_frame}-{end_frame} for length {length}"
                )
                continue

            for frame_index in range(start_frame, end_frame + 1):
                if coverage[frame_index] is not None:
                    errors.append(f"episode {episode_index}: overlapping label at frame {frame_index}")
                coverage[frame_index] = subtask

        missing = [idx for idx, label in enumerate(coverage) if label is None]
        if missing:
            errors.append(
                f"episode {episode_index}: missing labels for {len(missing)} frames "
                f"(first={missing[0]}, last={missing[-1]})"
            )

        counts.update(label for label in coverage if label is not None)

    unknown_episodes = set(segments) - set(episode_lengths)
    errors.extend([f"labels reference unknown merged episode {episode_index}" for episode_index in sorted(unknown_episodes)])

    if errors:
        raise ValueError("\n".join(errors[:20]))

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--merged-repo", default=DEFAULT_MERGED_REPO)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()

    merged_root = args.data_root / args.merged_repo
    episode_lengths = load_episode_lengths(merged_root)
    segments, rows = load_segments(args.labels)
    counts = validate_coverage(episode_lengths, segments)

    print(f"labels: {args.labels}")
    print(f"merged dataset: {merged_root}")
    print(f"rows: {rows}")
    print(f"episodes: {len(episode_lengths)}")
    print("frame counts:")
    for subtask, prompt in SUBTASK_PROMPTS.items():
        print(f"  {subtask}: {counts[subtask]}  ({prompt})")
    print("coverage: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
