#!/usr/bin/env python3
"""Real-time HIERARCHICAL driver: Qwen3-VL high level -> pi0.5 low level.

Extends eval_real_RTC.py: every control step pushes the cam_high frame into a
RemoteHighLevel (memer-style step() contract, model served remotely by
Qwen3-VL/qwen-vl-finetune/tools/serve_subtask_predictor.py); the predicted
subtask maps to the pi0.5 language prompt. The async RTC machinery is kept
verbatim; the prompt is re-read every step and chunks computed for a stale
prompt are discarded when they arrive.

Three decoupled rates -- the high level NEVER blocks the 30Hz control loop:
  control tick   30Hz   consumes the in-memory action chunk
  pi0.5 infer    async  RTC background request with action_prefix
  Qwen infer     async  RemoteHighLevel background thread (~0.3-1s/pred)

Robot usage (robot workstation):
    python examples/trossen_ai/eval_real_hierarchical.py \
        --policy_host iris-hgx-1 --subtask_host iris-hgx-1 --test

Dry run on the cluster, no robot (replays a LeRobot episode's frames/states):
    export HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data
    uv run examples/trossen_ai/eval_real_hierarchical.py \
        --policy_host localhost --subtask_host localhost \
        --replay-episode 27 --test

Per-episode logs (states/actions/subtask timeline) are written to
real_runs/<timestamp>/ for post-hoc analysis with the offline figure tooling.
"""

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import math
import pathlib
import sys
import threading
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy
import openpi_client.image_tools as image_tools

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hierarchical_highlevel import RemoteHighLevel  # noqa: E402

LEROBOT_FORK_PATH = "/home/iris/lerobot"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 25
CONTROL_FREQ = 15
D_EST = 10
FORCE_PROMPT_DEFAULT = "put the yellow block to the plate"


@dataclasses.dataclass
class PendingChunk:
    future: concurrent.futures.Future
    start_time: float
    d_est: int
    prompt: str


# --------------------------------------------------------------------------- #
# Observation sources: the real robot, or a LeRobot episode replay (dry run).
# --------------------------------------------------------------------------- #
class RobotSource:
    """Wraps the Trossen AI Solo robot (imports the lerobot fork lazily)."""

    def __init__(self):
        if LEROBOT_FORK_PATH not in sys.path:
            sys.path.insert(0, LEROBOT_FORK_PATH)
        from lerobot.common.robot_devices.robots.configs import TrossenAISoloRobotConfig
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config
        import torch

        self._torch = torch
        robot_config = TrossenAISoloRobotConfig(
            max_relative_target=None,
            min_time_to_move_multiplier=4.0,
            camera_interface="opencv",
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        self.cameras = list(robot_config.cameras.keys())
        self.highlevel_camera = "cam_high"
        logger.info(f"Robot connected. Cameras: {self.cameras}")

    def capture(self) -> dict:
        """Returns {"state": (D,), "images": {cam: CHW uint8 224}, "hl_frame": HWC uint8}."""
        obs = self.robot.capture_observation()
        state = obs["observation.state"].detach().cpu().numpy().astype(np.float32)
        images, hl_frame = {}, None
        for cam in self.cameras:
            img = obs[f"observation.images.{cam}"].detach().cpu().numpy()  # HWC uint8
            if cam == self.highlevel_camera:
                hl_frame = img.copy()
            small = image_tools.resize_with_pad(img, height=224, width=224)
            images[cam] = np.transpose(small, (2, 0, 1))
        if hl_frame is None:
            raise RuntimeError(f"high-level camera {self.highlevel_camera!r} missing")
        return {"state": state, "images": images, "hl_frame": hl_frame}

    def send_action(self, action: np.ndarray) -> None:
        self.robot.send_action(self._torch.from_numpy(np.asarray(action, dtype=np.float32)))

    def close(self) -> None:
        self.robot.disconnect()


class ReplayEpisodeSource:
    """Replays one LeRobot dataset episode's frames + states (no robot needed)."""

    def __init__(self, repo_id: str, episode_index: int):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id)
        self.dataset = LeRobotDataset(repo_id)
        edi = self.dataset.episode_data_index
        self._from = int(edi["from"][episode_index].item())
        self._to = int(edi["to"][episode_index].item())
        self._t = 0
        self.cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        self.highlevel_camera = "cam_high"
        logger.info(
            f"Replaying {repo_id} episode {episode_index}: "
            f"{self._to - self._from} frames @ {meta.fps}fps")

    def __len__(self) -> int:
        return self._to - self._from

    @property
    def exhausted(self) -> bool:
        return self._t >= len(self)

    @staticmethod
    def _to_hwc_uint8(img) -> np.ndarray:
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    def capture(self) -> dict:
        frame = self.dataset[self._from + min(self._t, len(self) - 1)]
        self._t += 1
        state = frame["observation.state"]
        if hasattr(state, "numpy"):
            state = state.numpy()
        images, hl_frame = {}, None
        for cam in self.cameras:
            img = self._to_hwc_uint8(frame[f"observation.images.{cam}"])
            if cam == self.highlevel_camera:
                hl_frame = img
            small = image_tools.resize_with_pad(img, height=224, width=224)
            images[cam] = np.transpose(small, (2, 0, 1))
        return {"state": np.asarray(state, dtype=np.float32), "images": images,
                "hl_frame": hl_frame}

    def send_action(self, action: np.ndarray) -> None:  # replay never moves anything
        pass

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
class HierarchicalRTCBridge:
    def __init__(
        self,
        source,
        *,
        policy_host: str,
        policy_port: int,
        subtask_host: str,
        subtask_port: int,
        control_frequency: int = CONTROL_FREQ,
        chunk_size: int = CHUNK_SIZE,
        d_est: int = D_EST,
        stale_margin: int = 2,
        late_tolerance: int = 4,
        max_action_delta: float = 0.25,
        history_seconds: float = 5.0,
        num_frames: int = 5,
        test_mode: bool = False,
        disable_rtc: bool = False,
        visualize: bool = True,
        force_prompt: str | None = None,
        log_dir: pathlib.Path | None = None,
    ):
        self.source = source
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.chunk_size = chunk_size
        self.d_est = d_est
        self.stale_margin = stale_margin
        self.late_tolerance = late_tolerance
        self.max_action_delta = max_action_delta
        self.test_mode = test_mode
        self.disable_rtc = disable_rtc
        self.visualize = visualize
        self.force_prompt = force_prompt
        if force_prompt is not None:
            logger.info(f"Low-level prompt FORCED to {force_prompt!r}; high level still runs "
                        "for display but does not drive the prompt.")
        self._viz_window = "high cam (subtask)"
        if disable_rtc:
            logger.info("RTC disabled: plain blocking chunk inference "
                        "(execute full chunk, then request next; replan on prompt switch).")
        self.log_dir = log_dir

        logger.info(f"Connecting to pi0.5 policy server at {policy_host}:{policy_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host, port=policy_port)

        logger.info(f"Connecting to Qwen subtask server at {subtask_host}:{subtask_port}")
        self.highlevel = RemoteHighLevel(
            host=subtask_host, port=subtask_port,
            fps=control_frequency, history_seconds=history_seconds, num_frames=num_frames,
        )

        self.active_prompt: str | None = None
        self.current_action_chunk: np.ndarray | None = None
        self.action_chunk_idx = 0
        self.episode_step = 0
        self.last_action: np.ndarray | None = None
        self._late_rate_limit_steps = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending: PendingChunk | None = None
        self._request_lock = threading.Lock()
        self._holding_pending_logged = False
        self._latest_obs: dict | None = None

        # episode logs
        self._log_states: list[np.ndarray] = []
        self._log_actions: list[np.ndarray] = []
        self._log_slugs: list[str] = []
        self._log_pred_age: list[int] = []
        self._log_events: list[dict] = []

    # ----------------------------- RTC core (from eval_real_RTC.py) -------- #
    def _active_chunk_limit(self) -> int:
        if self.current_action_chunk is None:
            return 0
        return min(self.chunk_size, len(self.current_action_chunk))

    def replan(self) -> None:
        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self._pending = None
        self._holding_pending_logged = False

    def _policy_obs(self, prompt: str, *, action_prefix=None, prefix_length=None) -> dict:
        obs = self._latest_obs
        result = {"state": obs["state"], "images": obs["images"], "prompt": prompt}
        if action_prefix is not None and prefix_length is not None and prefix_length > 0:
            result["action_prefix"] = action_prefix[:prefix_length].astype(np.float32)
            result["prefix_length"] = np.asarray(prefix_length, dtype=np.int32)
        return result

    def request_chunk(self, prompt: str, *, action_prefix=None, prefix_length=None) -> np.ndarray:
        observation = self._policy_obs(prompt, action_prefix=action_prefix,
                                       prefix_length=prefix_length)
        t0 = time.monotonic()
        with self._request_lock:
            response = self.policy_client.infer(observation)
        elapsed = (time.monotonic() - t0) * 1000.0
        chunk = np.asarray(response["actions"], dtype=np.float32)
        logger.info(f"Inference: chunk={chunk.shape}  {elapsed:.0f}ms  "
                    f"prefix={prefix_length or 0}  prompt={prompt!r}")
        return chunk

    def _launch_async_request(self, prompt: str) -> None:
        if self._pending is not None or self.current_action_chunk is None:
            return
        start_idx = self.action_chunk_idx
        if start_idx <= 0:
            return
        chunk_limit = self._active_chunk_limit()
        prefix = self.current_action_chunk[start_idx: min(start_idx + self.d_est, chunk_limit)]
        if len(prefix) == 0:
            return
        future = self._executor.submit(
            self.request_chunk, prompt, action_prefix=prefix, prefix_length=len(prefix))
        self._pending = PendingChunk(future=future, start_time=time.monotonic(),
                                     d_est=len(prefix), prompt=prompt)

    def _maybe_accept_async_chunk(self) -> None:
        if self._pending is None or not self._pending.future.done():
            return
        pending = self._pending
        self._pending = None
        self._holding_pending_logged = False
        finish_time = time.monotonic()
        try:
            new_chunk = np.asarray(pending.future.result(), dtype=np.float32)
        except Exception:
            logger.exception("Async RTC inference failed; keeping current chunk.")
            return

        # Hierarchical addition: a chunk computed for a prompt that has since
        # changed is useless -- discard it; the launch loop immediately requests
        # a fresh chunk with the new prompt.
        if pending.prompt != self.active_prompt:
            logger.info(f"RTC chunk discarded (stale prompt {pending.prompt!r} -> "
                        f"{self.active_prompt!r})")
            self._log_events.append({"step": self.episode_step, "event": "discard_stale_prompt"})
            return

        d_actual = math.ceil((finish_time - pending.start_time) / self.dt)
        new_chunk_limit = min(self.chunk_size, len(new_chunk))
        if d_actual <= pending.d_est and d_actual < new_chunk_limit:
            self.current_action_chunk = new_chunk
            self.action_chunk_idx = d_actual
            self._log_events.append({"step": self.episode_step, "event": "chunk_accept",
                                     "d_actual": d_actual})
            return
        if d_actual <= pending.d_est + self.late_tolerance and d_actual < new_chunk_limit - self.stale_margin:
            self.current_action_chunk = new_chunk
            self.action_chunk_idx = d_actual
            self._late_rate_limit_steps = min(self.late_tolerance, new_chunk_limit - d_actual)
            self._log_events.append({"step": self.episode_step, "event": "chunk_accept_late",
                                     "d_actual": d_actual})
            return
        logger.warning(f"RTC chunk discarded as stale: d_est={pending.d_est}, "
                       f"d_actual={d_actual}, horizon={new_chunk_limit}")
        self._log_events.append({"step": self.episode_step, "event": "chunk_discard_late"})

    def _hold_action_while_pending(self) -> np.ndarray:
        if not self._holding_pending_logged:
            logger.warning("Chunk exhausted while RTC request pending; holding last action.")
            self._holding_pending_logged = True
        if self.last_action is not None:
            return self.last_action.copy()
        chunk_limit = self._active_chunk_limit()
        if self.current_action_chunk is not None and chunk_limit > 0:
            return self.current_action_chunk[chunk_limit - 1].copy()
        raise RuntimeError("No action available to hold while waiting for RTC inference.")

    def _rate_limit(self, action: np.ndarray) -> np.ndarray:
        if self.last_action is None:
            return action
        delta = np.clip(action - self.last_action, -self.max_action_delta, self.max_action_delta)
        return (self.last_action + delta).astype(np.float32)

    def execute_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        self.last_action = action
        if self.test_mode:
            return
        self.source.send_action(action)

    def gravity_compensation_warmup(self, duration: float = 3.0) -> None:
        """Enable gravity comp so the user can hand-place the follower arm before rollout."""
        if duration <= 0:
            return
        robot = getattr(self.source, "robot", None)
        if robot is None:  # replay source has no real arms
            logger.info("No real robot; skipping gravity-compensation warm-up.")
            return

        logger.info(f"Gravity compensation enabled for {duration:.1f}s. "
                    "Move the follower arm to the desired initial pose.")
        for arm in robot.follower_arms.values():
            # In the Trossen LeRobot fork, Torque_Enable=0 switches the arm to
            # external_effort mode (zero external effort); the driver still
            # compensates for gravity and friction so the arm is hand-movable.
            arm.write("Torque_Enable", 0)

        end_time = time.monotonic() + duration
        last_seconds = None
        while time.monotonic() < end_time:
            remaining = max(0, int(math.ceil(end_time - time.monotonic())))
            if remaining != last_seconds:
                logger.info(f"Choose initial pose: {remaining}s remaining")
                last_seconds = remaining
            time.sleep(0.05)

        logger.info("Locking the selected initial pose...")
        for name, arm in robot.follower_arms.items():
            current_position = arm.read("Present_Position").astype(np.float32)
            arm.write("Torque_Enable", 1)
            arm.write("Goal_Position", current_position)
            logger.info(f"Locked {name} follower arm at: "
                        f"{np.round(current_position, decimals=4).tolist()}")

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0) -> None:
        from scipy.interpolate import PchipInterpolator

        current = self.source.capture()["state"]
        interpolator = PchipInterpolator(
            np.array([0.0, duration]), np.array([current, goal_position]), axis=0)
        t_start = time.time()
        while time.time() - t_start < duration:
            self.execute_action(interpolator(time.time() - t_start).astype(np.float32))
            time.sleep(self.dt)

    # ----------------------------- hierarchical step ----------------------- #
    def _step_once(self, *, continuous_mode: bool) -> np.ndarray:
        """Capture obs, update high level, manage prompt switches, pick action."""
        obs = self.source.capture()
        self._latest_obs = obs

        slug = self.highlevel.step(obs["hl_frame"])
        prompt = self.force_prompt if self.force_prompt is not None \
            else self.highlevel.lowlevel_prompt(slug)
        if self.active_prompt is None:
            self.active_prompt = prompt
            logger.info(f"Initial subtask: {slug!r} -> prompt {prompt!r}"
                        + (" (FORCED)" if self.force_prompt is not None else ""))
        elif prompt != self.active_prompt:
            logger.info(f"PROMPT SWITCH at step {self.episode_step}: "
                        f"{self.active_prompt!r} -> {prompt!r}")
            self._log_events.append({"step": self.episode_step, "event": "prompt_switch",
                                     "from": self.active_prompt, "to": prompt})
            self.active_prompt = prompt
            # In-flight request (if any) now carries a stale prompt; it will be
            # discarded on arrival and the launch loop below re-requests with
            # the new prompt + prefix continuity from the executing chunk.
            # Without RTC there is no in-flight request, so drop the stale chunk
            # outright and re-request synchronously with the new prompt.
            if self.disable_rtc:
                self.replan()

        self._render_highlevel(slug, obs)

        if not self.disable_rtc:
            self._maybe_accept_async_chunk()
        if self.current_action_chunk is None or self.action_chunk_idx >= self._active_chunk_limit():
            if not self.disable_rtc and self._pending is not None:
                action = self._hold_action_while_pending()
                self._record(obs, action, slug)
                return action
            self.current_action_chunk = self.request_chunk(self.active_prompt)
            self.action_chunk_idx = 0
            self._holding_pending_logged = False

        if continuous_mode and not self.disable_rtc:
            self._launch_async_request(self.active_prompt)
            self._maybe_accept_async_chunk()

        action = self.current_action_chunk[self.action_chunk_idx]
        self.action_chunk_idx += 1
        if self._late_rate_limit_steps > 0:
            action = self._rate_limit(action)
            self._late_rate_limit_steps -= 1
        self._record(obs, action, slug)
        return action

    @staticmethod
    def _to_hwc(img: np.ndarray) -> np.ndarray:
        """Coerce an image to HWC uint8 (handles CHW and float inputs)."""
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255.0).clip(0, 255)
        return np.ascontiguousarray(img.astype(np.uint8))

    @staticmethod
    def _tile_row(images: list, labels: list, tile: int = 224) -> np.ndarray | None:
        """Resize each image to tile x tile, label it, and hconcat into one row."""
        cells = []
        for img, label in zip(images, labels):
            cell = cv2.resize(HierarchicalRTCBridge._to_hwc(img), (tile, tile))
            cv2.putText(cell, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(cell, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cells.append(cell)
        return np.hstack(cells) if cells else None

    def _render_highlevel(self, slug: str, obs: dict) -> None:
        """Show everything fed to the models: high-level history window + low-level images."""
        if not self.visualize:
            return

        # High-level row: the exact frames most recently sent to the Qwen model.
        win = self.highlevel.last_window
        if win is not None:
            frames, idx = win
            hl_row = self._tile_row(list(frames), [f"HL t={i}" for i in idx])
        else:
            hl_row = self._tile_row([obs["hl_frame"]], ["HL (warming up)"])

        # Low-level row: the 3 camera images fed to pi0.5 (CHW 224 -> HWC).
        ll_imgs = list(obs["images"].values())
        ll_labels = [f"LL {cam}" for cam in obs["images"]]
        ll_row = self._tile_row(ll_imgs, ll_labels)

        rows = [r for r in (hl_row, ll_row) if r is not None]
        if not rows:
            return
        width = max(r.shape[1] for r in rows)
        rows = [r if r.shape[1] == width
                else np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
                for r in rows]

        age = self.highlevel.status.age_steps
        header = np.zeros((96, width, 3), np.uint8)
        if self.force_prompt is not None:
            subtask_line = f"subtask(pred): {slug}  [FORCED]"
            prompt_line = f"prompt: {self.force_prompt}"
        else:
            subtask_line = f"subtask: {slug}"
            prompt_line = f"prompt: {self.highlevel.lowlevel_prompt(slug)}"
        lines = [subtask_line, prompt_line,
                 f"step {self.episode_step}  age {age}  preds {self.highlevel.status.num_predictions}"]
        y = 26
        for text in lines:
            cv2.putText(header, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1, cv2.LINE_AA)
            y += 26

        canvas = np.vstack([header, *rows])
        try:
            cv2.imshow(self._viz_window, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except cv2.error:  # headless / no display available
            logger.warning("No display available; disabling visualization.")
            self.visualize = False

    def _record(self, obs: dict, action: np.ndarray, slug: str) -> None:
        self._log_states.append(np.asarray(obs["state"], dtype=np.float32))
        self._log_actions.append(np.asarray(action, dtype=np.float32))
        self._log_slugs.append(slug)
        self._log_pred_age.append(self.highlevel.status.age_steps)

    # ----------------------------- run loops ------------------------------- #
    def run_replay(self) -> None:
        """Continuous dry run over a replay source (no keyboard, no robot)."""
        logger.info(f"Replay dry run: {len(self.source)} frames at {self.control_frequency}Hz")
        self.highlevel.reset()
        t0 = time.monotonic()
        while not self.source.exhausted:
            action = self._step_once(continuous_mode=True)
            self.execute_action(action)
            self.episode_step += 1
            time.sleep(max(0.0, t0 + self.episode_step * self.dt - time.monotonic()))
        logger.info(f"Replay finished: {self.episode_step} steps, "
                    f"{self.highlevel.status.num_predictions} high-level predictions")

    def run_robot(self, gravity_comp_time: float = 3.0) -> None:
        """Interactive robot episode (keyboard controls from eval_real_RTC.py)."""
        import select
        import termios
        import tty

        self.gravity_compensation_warmup(duration=gravity_comp_time)

        class KeyReader:
            def __init__(self):
                self._fd = sys.stdin.fileno()
                self._old = None

            def __enter__(self):
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                return self

            def __exit__(self, *_):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

            def poll(self) -> str | None:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                return sys.stdin.read(1) if r else None

        print(f"\nHierarchical mode: prompt follows the Qwen subtask prediction.")
        print("Controls:  n=step  c=run continuously  r=replan  q=quit\n")
        self.highlevel.reset()
        is_first_step = True
        continuous_mode = False

        with KeyReader() as keys:
            while True:
                if continuous_mode:
                    ch = keys.poll()
                    if ch is not None:
                        key = ch.lower()
                        if key == "q":
                            return
                        if key == "r":
                            self.replan()
                            continue
                        continuous_mode = False
                        logger.info("Continuous mode stopped; back to single-step.")
                        continue
                else:
                    cmd = None
                    while cmd is None:
                        ch = keys.poll()
                        if ch is not None:
                            cmd = ch.lower()
                        else:
                            time.sleep(0.02)
                    if cmd == "q":
                        return
                    if cmd == "r":
                        self.replan()
                        continue
                    if cmd == "c":
                        continuous_mode = True
                        self.replan()
                        logger.info("Continuous hierarchical mode started.")
                        continue
                    if cmd != "n":
                        print(f"  unknown key {cmd!r}; valid: n c r q")
                        continue

                action = self._step_once(continuous_mode=continuous_mode)
                if is_first_step:
                    logger.info("Smooth-moving to first predicted action (5s)...")
                    if not self.test_mode:
                        self.move_to_start_position(action, duration=5.0)
                    is_first_step = False
                else:
                    self.execute_action(action)
                self.episode_step += 1
                if continuous_mode:
                    time.sleep(self.dt)

    def save_logs(self) -> None:
        if self.log_dir is None or not self._log_states:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.log_dir / "episode_log.npz",
            states=np.stack(self._log_states),
            actions=np.stack(self._log_actions),
            pred_age_steps=np.asarray(self._log_pred_age, dtype=np.int32),
        )
        slugs = self._log_slugs
        segments = []
        start = 0
        for t in range(1, len(slugs) + 1):
            if t == len(slugs) or slugs[t] != slugs[start]:
                segments.append({"start": start, "end": t - 1, "subtask": slugs[start]})
                start = t
        (self.log_dir / "timeline.json").write_text(json.dumps({
            "control_frequency": self.control_frequency,
            "chunk_size": self.chunk_size,
            "num_steps": len(slugs),
            "subtask_segments": segments,
            "per_step_subtask": slugs,
            "events": self._log_events,
            "num_highlevel_predictions": self.highlevel.status.num_predictions,
        }, indent=2))
        logger.info(f"Wrote episode logs to {self.log_dir}")

    def cleanup(self) -> None:
        self.highlevel.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.visualize:
            cv2.destroyAllWindows()
        self.save_logs()
        self.source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_host", default="iris-hgx-1")
    parser.add_argument("--policy_port", type=int, default=8000)
    parser.add_argument("--subtask_host", default="iris-hgx-1")
    parser.add_argument("--subtask_port", type=int, default=8001)
    parser.add_argument("--control_freq", type=int, default=CONTROL_FREQ)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--d_est", type=int, default=D_EST)
    parser.add_argument("--late_tolerance", type=int, default=4)
    parser.add_argument("--stale_margin", type=int, default=2)
    parser.add_argument("--max_action_delta", type=float, default=0.25)
    parser.add_argument("--history_seconds", type=float, default=5.0)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--replay-episode", type=int, default=None,
                        help="Dry run: replay this LeRobot episode instead of the robot.")
    parser.add_argument("--replay-repo-id", default="0528_merge_block_mem")
    parser.add_argument("--test", action="store_true",
                        help="Don't move the robot; log what would have been sent.")
    parser.add_argument("--visual_only", action="store_true",
                        help="Verify the pipeline (capture + high level + inference + "
                             "visualization) without ever sending actions to the robot.")
    parser.add_argument("--disable_rtc", action="store_true",
                        help="Turn off async real-time chunking; use plain blocking "
                             "chunk inference (replan on subtask/prompt switch).")
    parser.add_argument("--gravity_comp_time", type=float, default=3.0,
                        help="Seconds of gravity compensation at start for hand-placing "
                             "the follower arm at the initial pose (robot only).")
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable the high-cam window with the subtask overlay.")
    parser.add_argument("--force_prompt", default=None, nargs="?",
                        const=FORCE_PROMPT_DEFAULT,
                        help="Pin the low-level prompt (high level still runs for display but "
                             f"does not drive it). Bare flag defaults to {FORCE_PROMPT_DEFAULT!r}; "
                             "pass a string to override.")
    parser.add_argument("--log-dir", default=None,
                        help="Default: real_runs/<timestamp>/")
    args = parser.parse_args()

    if args.replay_episode is not None:
        source = ReplayEpisodeSource(args.replay_repo_id, args.replay_episode)
        test_mode = True
        default_log = f"real_runs/replay_ep{args.replay_episode}_{time.strftime('%m%d_%H%M%S')}"
    else:
        source = RobotSource()
        test_mode = args.test or args.visual_only
        default_log = f"real_runs/robot_{time.strftime('%m%d_%H%M%S')}"

    bridge = HierarchicalRTCBridge(
        source,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        subtask_host=args.subtask_host,
        subtask_port=args.subtask_port,
        control_frequency=args.control_freq,
        chunk_size=args.chunk_size,
        d_est=args.d_est,
        stale_margin=args.stale_margin,
        late_tolerance=args.late_tolerance,
        max_action_delta=args.max_action_delta,
        history_seconds=args.history_seconds,
        num_frames=args.num_frames,
        test_mode=test_mode,
        disable_rtc=args.disable_rtc,
        visualize=not args.no_viz,
        force_prompt=args.force_prompt,
        log_dir=pathlib.Path(args.log_dir or default_log),
    )
    try:
        if args.replay_episode is not None:
            bridge.run_replay()
        else:
            bridge.run_robot(gravity_comp_time=args.gravity_comp_time)
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
