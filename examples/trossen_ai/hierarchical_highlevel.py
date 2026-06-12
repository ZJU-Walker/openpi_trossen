"""Driver-side high-level policy wrapper (memer-style contract).

Runs on the robot workstation; depends only on numpy + openpi_client. The Qwen
model itself runs remotely behind tools/serve_subtask_predictor.py (Qwen3-VL
repo) speaking the openpi websocket protocol.

Contract (mirrors memer's MemERDeploymentPolicy):

    hl = RemoteHighLevel(host="iris-hgx-1", port=8001)
    hl.reset()
    for each 30Hz control step:
        slug = hl.step(cam_high_frame)   # NON-BLOCKING, returns latest prediction

step() pushes the frame into a ring buffer and returns the latest COMPLETED
subtask prediction; a single background thread continuously builds the same
5s-history / 5-frame / front-padded window the model was trained on, sends it
to the server, and updates the current prediction at the model's natural rate
(~0.3-1s per prediction). The control loop is never blocked by the high level.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from openpi_client import websocket_client_policy

logger = logging.getLogger(__name__)

# Same map as the offline eval + training config; the server also advertises it
# in its connect metadata (used here as the source of truth when available).
DEFAULT_PROMPT_MAP = {
    "observe_human": "wait and observe the human",
    "put_green_block_to_plate": "put the green block to the plate",
    "put_yellow_block_to_plate": "put the yellow block to the plate",
}


def sampled_indices(clip_start: int, clip_end: int, nframes: int) -> list[int]:
    if nframes <= 1:
        return [clip_start]
    return [int(round(clip_start + (clip_end - clip_start) * k / (nframes - 1)))
            for k in range(nframes)]


def window_frame_indices(t: int, *, num_frames: int, history_frames: int) -> list[int]:
    clip_start = max(0, t - history_frames + 1)
    idx = sampled_indices(clip_start, t, min(num_frames, t - clip_start + 1))
    if len(idx) < num_frames:
        idx = [idx[0]] * (num_frames - len(idx)) + idx
    return idx


@dataclass
class HighLevelStatus:
    subtask: str
    lowlevel_prompt: str
    prediction_timestep: int    # control timestep the prediction was made FOR
    age_steps: int              # current_timestep - prediction_timestep
    num_predictions: int
    last_infer_ms: float | None


class RemoteHighLevel:
    def __init__(
        self,
        host: str,
        port: int = 8001,
        *,
        fps: float = 30.0,
        history_seconds: float = 5.0,
        num_frames: int = 5,
        default_subtask: str = "observe_human",
        prompt_map: dict[str, str] | None = None,
    ):
        self._fps = fps
        self._history_frames = max(2, int(round(history_seconds * fps)))
        self._num_frames = num_frames
        self._default_subtask = default_subtask

        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        meta = self._client.get_server_metadata()
        self._prompt_map = dict(prompt_map or meta.get("prompt_map") or DEFAULT_PROMPT_MAP)
        if default_subtask not in self._prompt_map:
            raise ValueError(f"default_subtask {default_subtask!r} not in prompt map")
        # Fail fast if the server's window rule differs from ours.
        for key, ours in (("history_seconds", history_seconds),
                          ("num_frames", num_frames), ("fps", fps)):
            theirs = meta.get(key)
            if theirs is not None and float(theirs) != float(ours):
                logger.warning(f"Server {key}={theirs} != driver {key}={ours}")

        self._lock = threading.Lock()
        self._frames: dict[int, np.ndarray] = {}   # timestep -> frame (ring buffer)
        self._timestep = -1
        self._subtask = default_subtask
        self._prediction_timestep = -1
        self._num_predictions = 0
        self._last_infer_ms: float | None = None
        self._last_window: tuple[np.ndarray, list[int]] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear state and (re)start the background prediction thread."""
        self.stop()
        with self._lock:
            self._frames = {}
            self._timestep = -1
            self._subtask = self._default_subtask
            self._prediction_timestep = -1
            self._num_predictions = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._predict_loop, daemon=True,
                                        name="highlevel-predict")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def step(self, frame: np.ndarray) -> str:
        """Push the newest cam_high frame (H,W,C uint8); return latest subtask.

        Non-blocking: returns the latest completed prediction (or the default
        subtask before the first prediction lands).
        """
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        with self._lock:
            self._timestep += 1
            self._frames[self._timestep] = frame
            oldest_needed = self._timestep - self._history_frames + 1
            for t in [t for t in self._frames if t < oldest_needed]:
                del self._frames[t]
            return self._subtask

    @property
    def status(self) -> HighLevelStatus:
        with self._lock:
            return HighLevelStatus(
                subtask=self._subtask,
                lowlevel_prompt=self._prompt_map[self._subtask],
                prediction_timestep=self._prediction_timestep,
                age_steps=self._timestep - self._prediction_timestep,
                num_predictions=self._num_predictions,
                last_infer_ms=self._last_infer_ms,
            )

    def lowlevel_prompt(self, slug: str | None = None) -> str:
        with self._lock:
            return self._prompt_map[slug if slug is not None else self._subtask]

    @property
    def last_window(self) -> tuple[np.ndarray, list[int]] | None:
        """The exact (frames, frame_indices) most recently sent to the model."""
        with self._lock:
            if self._last_window is None:
                return None
            frames, idx = self._last_window
            return frames.copy(), list(idx)

    # ------------------------------------------------------------------ #
    def _snapshot_window(self) -> tuple[np.ndarray, list[int]] | None:
        with self._lock:
            t = self._timestep
            if t < 0:
                return None
            idx = window_frame_indices(t, num_frames=self._num_frames,
                                       history_frames=self._history_frames)
            try:
                frames = np.stack([self._frames[i] for i in idx])
            except KeyError:
                return None  # buffer was trimmed mid-build; retry next loop
        return frames, idx

    def _predict_loop(self) -> None:
        while not self._stop.is_set():
            snap = self._snapshot_window()
            if snap is None:
                time.sleep(0.05)
                continue
            frames, idx = snap
            target_timestep = idx[-1]
            with self._lock:
                self._last_window = (frames, idx)
            try:
                result = self._client.infer({
                    "frames": frames,
                    "frame_indices": np.asarray(idx, dtype=np.int32),
                    "fps": np.float32(self._fps),
                })
            except Exception as e:  # transient network/server hiccup: keep last subtask
                logger.warning(f"High-level inference failed (keeping last subtask): {e}")
                time.sleep(0.5)
                continue

            with self._lock:
                self._num_predictions += 1
                self._last_infer_ms = result.get("server_timing", {}).get("infer_ms")
                if result.get("valid") and result["subtask"] in self._prompt_map:
                    if result["subtask"] != self._subtask:
                        logger.info(
                            f"Subtask switch: {self._subtask} -> {result['subtask']} "
                            f"(predicted at t={target_timestep}, now t={self._timestep})")
                    self._subtask = result["subtask"]
                    self._prediction_timestep = target_timestep
                else:
                    logger.warning(f"Ignoring invalid subtask {result.get('subtask')!r}")
