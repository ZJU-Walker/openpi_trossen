#!/usr/bin/env python3
"""
Trossen AI Solo <-> OpenPI RTC policy-server bridge.

This variant keeps an asynchronous inference request in flight during continuous
rollouts. The request includes an action_prefix from the current chunk, so the
server-side pi0/pi0.5 policy can condition on committed actions and produce a
chunk whose already-past prefix remains fixed.

Usage:
    python examples/trossen_ai/eval_real_RTC.py --policy_host iris-hgx-1 --test
"""

import argparse
import concurrent.futures
import dataclasses
import logging
import select
import sys
import termios
import threading
import time
import tty

LEROBOT_FORK_PATH = "/home/iris/lerobot"
if LEROBOT_FORK_PATH and LEROBOT_FORK_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_FORK_PATH)

import cv2  # noqa: E402
from lerobot.common.robot_devices.robots.configs import TrossenAISoloRobotConfig  # noqa: E402
from lerobot.common.robot_devices.robots.utils import make_robot_from_config  # noqa: E402
import numpy as np  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402
import openpi_client.image_tools as image_tools  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_IP = "iris-hgx-1"
PROMPT = "put the green block to the plate"
D_EST = 10
CHUNK_SIZE = 50
CONTROL_FREQ = 30  # must match the training data fps (0528_merge_block_mem is 30fps)


@dataclasses.dataclass
class PendingChunk:
    future: concurrent.futures.Future
    start_time: float
    d_est: int
    start_idx: int


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


class TrossenOpenPIRTCBridge:
    def __init__(
        self,
        policy_server_host: str = SERVER_IP,
        policy_server_port: int = 8000,
        control_frequency: int = CONTROL_FREQ,
        chunk_size: int = CHUNK_SIZE,
        d_est: int = D_EST,
        stale_margin: int = 2,
        late_tolerance: int = 4,
        max_action_delta: float = 0.25,
        *,
        test_mode: bool = False,
    ):
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.chunk_size = chunk_size
        self.d_est = d_est
        self.stale_margin = stale_margin
        self.late_tolerance = late_tolerance
        self.max_action_delta = max_action_delta
        self.test_mode = test_mode

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )

        robot_config = TrossenAISoloRobotConfig(
            max_relative_target=None,
            min_time_to_move_multiplier=4.0,
            camera_interface="opencv",
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()

        self.cameras = list(robot_config.cameras.keys())
        self.action_dim = sum(len(b.motor_names) for b in self.robot.follower_arms.values())
        logger.info(f"Cameras: {self.cameras}  action_dim: {self.action_dim}")

        self.current_action_chunk: np.ndarray | None = None
        self.action_chunk_idx = 0
        self.episode_step = 0
        self.last_action: np.ndarray | None = None
        self._late_rate_limit_steps = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending: PendingChunk | None = None
        self._request_lock = threading.Lock()
        self._holding_pending_logged = False

    def gravity_compensation_warmup(self, duration: float = 5.0) -> None:
        """Allow the user to manually place the follower arm before rollout."""
        if duration <= 0:
            return

        if self.test_mode:
            logger.info("TEST MODE: skipping gravity-compensation warm-up.")
            return

        logger.info(
            f"Gravity compensation enabled for {duration:.1f}s. "
            "Move the follower arm to the desired initial pose."
        )
        for arm in self.robot.follower_arms.values():
            # In the Trossen LeRobot fork, Torque_Enable=0 switches the arm to
            # external_effort mode and sends zero external effort. The driver
            # still compensates for gravity and friction in this mode.
            arm.write("Torque_Enable", 0)

        end_time = time.monotonic() + duration
        last_seconds = None
        while time.monotonic() < end_time:
            remaining = max(0, int(np.ceil(end_time - time.monotonic())))
            if remaining != last_seconds:
                logger.info(f"Choose initial pose: {remaining}s remaining")
                last_seconds = remaining
            time.sleep(0.05)

        logger.info("Locking the selected initial pose...")
        for name, arm in self.robot.follower_arms.items():
            current_position = arm.read("Present_Position").astype(np.float32)
            arm.write("Torque_Enable", 1)
            arm.write("Goal_Position", current_position)
            logger.info(
                f"Locked {name} follower arm at: "
                f"{np.round(current_position, decimals=4).tolist()}"
            )

    def _active_chunk_limit(self) -> int:
        if self.current_action_chunk is None:
            return 0
        return min(self.chunk_size, len(self.current_action_chunk))

    def replan(self) -> None:
        """Discard the active chunk and any in-flight RTC request.

        The next `_next_action()` call will synchronously request a fresh chunk.
        """
        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self._pending = None
        self._holding_pending_logged = False

    def get_observation(
        self,
        task_prompt: str,
        *,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
    ) -> dict:
        obs = self.robot.capture_observation()
        state = obs["observation.state"].detach().cpu().numpy().astype(np.float32)

        images = {}
        for cam in self.cameras:
            img = obs[f"observation.images.{cam}"].detach().cpu().numpy()
            img = image_tools.resize_with_pad(img, height=224, width=224)
            cv2.imwrite(f"{cam}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            images[cam] = np.transpose(img, (2, 0, 1))

        result = {"state": state, "images": images, "prompt": task_prompt}
        if action_prefix is not None and prefix_length is not None and prefix_length > 0:
            result["action_prefix"] = action_prefix[:prefix_length].astype(np.float32)
            result["prefix_length"] = np.asarray(prefix_length, dtype=np.int32)
        return result

    def request_chunk(
        self,
        task_prompt: str,
        *,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
    ) -> np.ndarray:
        observation = self.get_observation(task_prompt, action_prefix=action_prefix, prefix_length=prefix_length)
        t0 = time.monotonic()
        with self._request_lock:
            response = self.policy_client.infer(observation)
        elapsed = (time.monotonic() - t0) * 1000.0
        chunk = np.asarray(response["actions"], dtype=np.float32)
        logger.info(f"Inference: chunk={chunk.shape}  {elapsed:.0f}ms  prefix={prefix_length or 0}")
        return chunk

    def _launch_async_request(self, task_prompt: str) -> None:
        if self._pending is not None or self.current_action_chunk is None:
            return

        start_idx = self.action_chunk_idx
        if start_idx <= 0:
            return

        chunk_limit = self._active_chunk_limit()
        prefix = self.current_action_chunk[start_idx : min(start_idx + self.d_est, chunk_limit)]
        if len(prefix) == 0:
            return

        start_time = time.monotonic()
        future = self._executor.submit(
            self.request_chunk,
            task_prompt,
            action_prefix=prefix,
            prefix_length=len(prefix),
        )
        self._pending = PendingChunk(
            future=future, start_time=start_time, d_est=len(prefix), start_idx=start_idx
        )
        logger.info(f"RTC request launched: start_idx={start_idx} d_est={len(prefix)}")

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

        # Count actions actually consumed since the request was launched; the new
        # chunk's prefix is conditioned on old_chunk[start_idx:], so new_chunk[d_actual]
        # is exactly the next action in the committed timeline. (Wall-clock ceil
        # systematically overshoots by 1 because acceptance polls at step boundaries,
        # skipping one action per replan.)
        d_actual = max(0, self.action_chunk_idx - pending.start_idx)
        elapsed_ms = (finish_time - pending.start_time) * 1000.0
        new_chunk_limit = min(self.chunk_size, len(new_chunk))
        if d_actual <= pending.d_est and d_actual < new_chunk_limit:
            self.current_action_chunk = new_chunk
            self.action_chunk_idx = d_actual
            logger.info(
                f"RTC chunk accepted: d_actual={d_actual}, switch_idx={self.action_chunk_idx}, "
                f"elapsed={elapsed_ms:.0f}ms"
            )
            return

        if d_actual <= pending.d_est + self.late_tolerance and d_actual < new_chunk_limit - self.stale_margin:
            self.current_action_chunk = new_chunk
            self.action_chunk_idx = d_actual
            self._late_rate_limit_steps = min(self.late_tolerance, new_chunk_limit - d_actual)
            logger.info(
                "RTC chunk accepted late with rate limiting: "
                f"d_est={pending.d_est}, d_actual={d_actual}, switch_idx={self.action_chunk_idx}"
            )
            return

        logger.warning(
            "RTC chunk discarded as stale: "
            f"d_est={pending.d_est}, d_actual={d_actual}, horizon={new_chunk_limit}"
        )

    def _hold_action_while_pending(self) -> np.ndarray:
        if not self._holding_pending_logged:
            logger.warning("Current chunk exhausted while RTC request is pending; holding last action.")
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
            logger.info(f"TEST MODE: would execute action: {action}")
            return
        self.robot.send_action(torch.from_numpy(action))

    def _next_action(self, task_prompt: str, *, continuous_mode: bool) -> np.ndarray:
        self._maybe_accept_async_chunk()
        if self.current_action_chunk is None or self.action_chunk_idx >= self._active_chunk_limit():
            if self._pending is not None:
                return self._hold_action_while_pending()
            self.current_action_chunk = self.request_chunk(task_prompt)
            self.action_chunk_idx = 0
            self._holding_pending_logged = False

        if continuous_mode:
            self._launch_async_request(task_prompt)
            self._maybe_accept_async_chunk()

        action = self.current_action_chunk[self.action_chunk_idx]
        self.action_chunk_idx += 1
        if self._late_rate_limit_steps > 0:
            action = self._rate_limit(action)
            self._late_rate_limit_steps -= 1
        return action

    def run_episode(self, task_prompt: str, gravity_comp_time: float = 5.0) -> None:
        self.gravity_compensation_warmup(duration=gravity_comp_time)

        print(f"\nPrompt: {task_prompt!r}\nChunk size: {self.chunk_size}  RTC d_est: {self.d_est}")
        print("Controls:  n=step  c=run continuously  r=replan  q=quit\n")

        continuous_mode = False

        with KeyReader() as keys:
            while True:
                if continuous_mode:
                    ch = keys.poll()
                    if ch is not None:
                        key = ch.lower()
                        if key == "q":
                            logger.info("Quit requested.")
                            return
                        if key == "r":
                            self.replan()
                            logger.info("Re-plan requested mid-continuous.")
                            continue
                        continuous_mode = False
                        logger.info("Continuous mode stopped; back to single-step.")
                        continue
                    cmd = "n"
                else:
                    cmd = None
                    while cmd is None:
                        ch = keys.poll()
                        if ch is not None:
                            cmd = ch.lower()
                        else:
                            time.sleep(0.02)

                    if cmd == "q":
                        logger.info("Quit requested.")
                        return
                    if cmd == "r":
                        self.replan()
                        logger.info("Re-plan requested; current chunk discarded.")
                        continue
                    if cmd == "c":
                        continuous_mode = True
                        self.replan()
                        logger.info("Continuous mode: async RTC replanning until interrupted.")
                        continue
                    if cmd != "n":
                        print(f"  unknown key {cmd!r}; valid: n c r q")
                        continue

                action = self._next_action(task_prompt, continuous_mode=continuous_mode)
                self.execute_action(action)

                self.episode_step += 1
                logger.info(f"step {self.episode_step}  chunk_idx={self.action_chunk_idx - 1}")

                if continuous_mode:
                    time.sleep(self.dt)

    def cleanup(self) -> None:
        logger.info("Cleaning up...")
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen AI Solo <-> OpenPI RTC eval.")
    parser.add_argument("--policy_host", default=SERVER_IP, help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=CONTROL_FREQ, help="Control frequency in Hz")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Nominal action chunk size.")
    parser.add_argument("--d_est", type=int, default=D_EST, help="Conservative prefix length sent to the policy.")
    parser.add_argument("--late_tolerance", type=int, default=4, help="Late frames tolerated with rate limiting.")
    parser.add_argument("--stale_margin", type=int, default=2, help="Discard chunks this close to horizon end.")
    parser.add_argument("--max_action_delta", type=float, default=0.25, help="Per-joint late-switch rate limit.")
    parser.add_argument("--task_prompt", default=PROMPT, help="Task description for the policy")
    parser.add_argument("--gravity_comp_time", type=float, default=5.0,
                        help="Seconds of gravity compensation for manually choosing the initial pose.")
    parser.add_argument("--test", action="store_true", help="Don't move the robot; log what would have been sent.")
    args = parser.parse_args()

    bridge = TrossenOpenPIRTCBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        chunk_size=args.chunk_size,
        d_est=args.d_est,
        stale_margin=args.stale_margin,
        late_tolerance=args.late_tolerance,
        max_action_delta=args.max_action_delta,
        test_mode=args.test,
    )
    try:
        bridge.run_episode(
            task_prompt=args.task_prompt,
            gravity_comp_time=args.gravity_comp_time,
        )
    finally:
        bridge.cleanup()
