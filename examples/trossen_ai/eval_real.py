#!/usr/bin/env python3
"""
Trossen AI Solo <-> OpenPI Policy Server Bridge (keyboard-controlled eval).

Mirror of main.py with three changes:
  - OpenCV cameras + TrossenAISoloRobotConfig (matches the data-collection
    setup used by `lerobot/scripts/control_robot_pedal.py
    --robot.type=trossen_ai_solo --robot.camera_interface=opencv`)
  - Keyboard control during rollout:
        n = execute next single action
        c = continuous: auto-step and auto-replan forever (any key drops back
            to single-step; r forces an immediate replan; q quits)
        r = re-plan now (discard current chunk)
        q = quit
  - --test runs everything but skips robot.send_action()

Usage:
    python eval_real.py --policy_host iris-hgx-1 --task_prompt "handover the marker"
"""

import argparse
import logging
import select
import sys
import termios
import time
import tty

# The fork at /home/iris/lerobot has trossen_ai_solo + opencv camera support.
# Insert ahead of any pip-installed lerobot so we pick up the same robot
# definition the data-collection script uses.
LEROBOT_FORK_PATH = "/home/iris/lerobot"
if LEROBOT_FORK_PATH and LEROBOT_FORK_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_FORK_PATH)

import cv2
from lerobot.common.robot_devices.robots.configs import TrossenAISoloRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
import numpy as np
from openpi_client import websocket_client_policy
from scipy.interpolate import PchipInterpolator
import torch
import openpi_client.image_tools as image_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_IP = "10.79.12.191"
PROMPT = "handover the marker"


class KeyReader:
    """Non-blocking single-character stdin reader (POSIX raw mode)."""

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


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Solo and the OpenPI policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 15,
        chunk_size: int = 25,
        test_mode: bool = False,
    ):
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.chunk_size = chunk_size
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

        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.episode_step = 0



    def get_observation(self, task_prompt: str) -> dict:
        """Capture observation and pack into the policy-side dict."""
        obs = self.robot.capture_observation()
        state = obs["observation.state"].detach().cpu().numpy().astype(np.float32)

        # OpenCV cameras already return RGB (color_mode="rgb" is the default),
        # so no BGR->RGB conversion is needed here.
        images = {}
        for cam in self.cameras:
            img = obs[f"observation.images.{cam}"].detach().cpu().numpy()
            # img = cv2.resize(img, (224, 224))
            img = image_tools.resize_with_pad(img, height=224, width=224)
            cv2.imwrite(f"{cam}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            images[cam] = np.transpose(img, (2, 0, 1))  # CHW

        return {"state": state, "images": images, "prompt": task_prompt}

    def execute_action(self, action: np.ndarray) -> None:
        if self.test_mode:
            logger.info(f"TEST MODE: would execute action: {action}")
            return
        self.robot.send_action(torch.from_numpy(np.asarray(action, dtype=np.float32)))

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0) -> None:
        """Smoothly PCHIP-interpolate from current pose to the policy's first action."""
        obs = self.robot.capture_observation()
        current = obs["observation.state"].detach().cpu().numpy().astype(np.float32)
        interpolator = PchipInterpolator(
            np.array([0.0, duration]), np.array([current, goal_position]), axis=0
        )
        t_start = time.time()
        while time.time() - t_start < duration:
            self.execute_action(interpolator(time.time() - t_start).astype(np.float32))
            time.sleep(self.dt)

    def request_chunk(self, task_prompt: str) -> np.ndarray:
        observation = self.get_observation(task_prompt)
        t0 = time.monotonic()
        response = self.policy_client.infer(observation)
        elapsed = (time.monotonic() - t0) * 1000.0
        chunk = np.asarray(response["actions"], dtype=np.float32)
        logger.info(f"Inference: chunk={chunk.shape}  {elapsed:.0f}ms")
        return chunk

    def run_episode(self, task_prompt: str) -> None:
        """Keyboard-controlled rollout. See module docstring for key bindings."""
        print(f"\nPrompt: {task_prompt!r}\nChunk size: {self.chunk_size}")
        print("Controls:  n=step  c=run continuously  r=replan  q=quit\n")

        is_first_step = True
        continuous_mode = False

        with KeyReader() as keys:
            while True:
                if continuous_mode:
                    # Non-blocking key poll: q quits, r forces a replan, any
                    # other key drops back to single-step mode.
                    ch = keys.poll()
                    if ch is not None:
                        key = ch.lower()
                        if key == "q":
                            logger.info("Quit requested.")
                            return
                        if key == "r":
                            self.current_action_chunk = None
                            self.action_chunk_idx = 0
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
                        self.current_action_chunk = None
                        self.action_chunk_idx = 0
                        logger.info("Re-plan requested; current chunk discarded.")
                        continue
                    if cmd == "c":
                        continuous_mode = True
                        self.current_action_chunk = None
                        self.action_chunk_idx = 0
                        logger.info("Continuous mode: auto-stepping and auto-replanning until 'q'.")
                        continue
                    if cmd != "n":
                        print(f"  unknown key {cmd!r}; valid: n c r q")
                        continue

                step_start = time.perf_counter()

                if self.current_action_chunk is None or self.action_chunk_idx >= self.chunk_size:
                    self.current_action_chunk = self.request_chunk(task_prompt)
                    self.action_chunk_idx = 0

                action = self.current_action_chunk[self.action_chunk_idx]
                self.action_chunk_idx += 1

                if is_first_step:
                    logger.info("Smooth-moving to first predicted action (5s)...")
                    self.move_to_start_position(action, duration=5.0)
                    is_first_step = False
                else:
                    self.execute_action(action)

                self.episode_step += 1
                logger.info(f"step {self.episode_step}  chunk_idx={self.action_chunk_idx - 1}")

                if continuous_mode:
                    time.sleep(1.0 / 30)  # hardcoded 20 Hz cadence

    def cleanup(self) -> None:
        logger.info("Cleaning up...")
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen AI Solo <-> OpenPI eval (keyboard-controlled).")
    parser.add_argument("--policy_host", default=SERVER_IP, help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument("--chunk_size", type=int, default=25,
                        help="Actions consumed per inference (and per `c` batch).")
    parser.add_argument("--task_prompt", default=PROMPT, help="Task description for the policy")
    parser.add_argument("--test", action="store_true",
                        help="Don't move the robot; log what would have been sent.")
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        chunk_size=args.chunk_size,
        test_mode=args.test,
    )
    try:
        bridge.run_episode(task_prompt=args.task_prompt)
    finally:
        bridge.cleanup()
