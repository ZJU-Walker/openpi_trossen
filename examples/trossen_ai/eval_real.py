#!/usr/bin/env python3
"""Real-robot evaluation client for OpenPI policy server, with keyboard control.

Adapted from main.py for our 3-camera Trossen setup (no cam_low).

Three keyboard-driven modes during a rollout:

    n  → execute ONE action from the current chunk (single-step)
    c  → execute CHUNK_SIZE actions, auto-replan after each batch, repeat
         (true autonomous rollout — equivalent to main.py behavior)
    r  → discard current chunk and force re-plan on next step
    q  → stop the episode cleanly

You can switch modes mid-episode. After every action, the script:
  - logs the consumed action + the joint state at that step
  - dumps a row to actions.csv
  - optionally saves cam_high frame as a JPG (--save-frames)

Usage:
    python eval_real.py \
        --policy-host iris-hgx-1 \
        --policy-port 8000 \
        --task-prompt "help a human pack a box" \
        --chunk-size 25

Test mode (no robot motion, save everything that would be sent + received):
    python eval_real.py --test --chunk-size 25
    # Implies --save-frames. Per-inference inputs/outputs land in
    # <output-dir>/server_inputs/infer_NNNN/ — see _request_chunk().

Notes:
  - Set ROBOT IPs and camera SERIALS below before running.
  - Policy server must already be running and have a checkpoint loaded:
        uv run scripts/serve_policy.py policy:checkpoint \
            --policy.config=pi05_trossen_pack_with_human \
            --policy.dir=checkpoints/pi05_trossen_pack_with_human/pi05_pack_with_human_0514/<step>
"""

import argparse
import csv
import logging
import pathlib
import select
import signal
import sys
import termios
import time
import tty
from datetime import datetime

import cv2
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np
from openpi_client import websocket_client_policy
from scipy.interpolate import PchipInterpolator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — edit these for your setup. CLI flags override.
# ---------------------------------------------------------------------------
POLICY_HOST = "iris-hgx-1"       # e.g. node hosting the OpenPI policy server
POLICY_PORT = 8000
TASK_PROMPT = "help a human pack a box"

LEFT_ARM_IP = "TODO_LEFT_IP"     # e.g. "192.168.1.5"
RIGHT_ARM_IP = "TODO_RIGHT_IP"   # e.g. "192.168.1.4"

CAM_HIGH_SERIAL = "TODO_CAM_HIGH_SERIAL"
CAM_LEFT_WRIST_SERIAL = "TODO_CAM_LEFT_WRIST_SERIAL"
CAM_RIGHT_WRIST_SERIAL = "TODO_CAM_RIGHT_WRIST_SERIAL"
# ---------------------------------------------------------------------------


# Camera key names MUST match the dataset and the policy's repack_transforms.
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class KeyReader:
    """Non-blocking single-character reader for stdin (POSIX raw mode)."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_attrs = None

    def __enter__(self):
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def poll(self) -> str | None:
        """Return one character if available, else None. Non-blocking."""
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None


class TrossenEvalBridge:
    """Keyboard-controlled real-robot eval client."""

    def __init__(
        self,
        policy_host: str,
        policy_port: int,
        task_prompt: str,
        chunk_size: int = 25,
        control_frequency: int = 30,
        save_frames_dir: pathlib.Path | None = None,
        output_dir: pathlib.Path | None = None,
        startup_warm_duration: float = 5.0,
        test_mode: bool = False,
    ):
        self.task_prompt = task_prompt
        self.chunk_size = chunk_size
        self.dt = 1.0 / control_frequency
        self.save_frames_dir = save_frames_dir
        self.startup_warm_duration = startup_warm_duration
        self.test_mode = test_mode

        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = pathlib.Path("eval_real_outputs") / ts
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        if save_frames_dir is None and self.save_frames_dir is None:
            pass  # explicit None means no frames
        logger.info(f"Logging to: {self.output_dir}")

        logger.info(f"Connecting to policy server at {policy_host}:{policy_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host, port=policy_port
        )

        for sentinel in (LEFT_ARM_IP, RIGHT_ARM_IP, CAM_HIGH_SERIAL,
                         CAM_LEFT_WRIST_SERIAL, CAM_RIGHT_WRIST_SERIAL):
            if sentinel.startswith("TODO_"):
                raise RuntimeError(
                    f"Placeholder {sentinel!r} still set. Edit eval_real.py to fill in "
                    "your arm IPs and camera serial numbers."
                )

        robot_config = BiWidowXAIFollowerRobotConfig(
            id="bimanual_follower",
            left_arm_ip_address=LEFT_ARM_IP,
            right_arm_ip_address=RIGHT_ARM_IP,
            min_time_to_move_multiplier=4.0,
            loop_rate=control_frequency,
            cameras={
                "cam_high": RealSenseCameraConfig(
                    serial_number_or_name=CAM_HIGH_SERIAL,
                    width=640, height=480, fps=30, use_depth=False,
                ),
                "cam_left_wrist": RealSenseCameraConfig(
                    serial_number_or_name=CAM_LEFT_WRIST_SERIAL,
                    width=640, height=480, fps=30, use_depth=False,
                ),
                "cam_right_wrist": RealSenseCameraConfig(
                    serial_number_or_name=CAM_RIGHT_WRIST_SERIAL,
                    width=640, height=480, fps=30, use_depth=False,
                ),
            },
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        logger.info("Robot connected.")

        self.action_chunk: np.ndarray | None = None
        self.chunk_idx = 0
        self.episode_step = 0
        self._inference_count = 0
        self._csv_path = self.output_dir / "actions.csv"
        self._csv_file = self._csv_path.open("w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_header_written = False

    # ----- robot I/O -----

    def _read_observation(self) -> dict:
        """Capture observation, transform images CHW RGB uint8, return policy-ready dict."""
        raw = self.robot.get_observation()

        joint_pos_keys = [k for k in raw.keys() if k.endswith(".pos")]
        joint_positions = np.array([raw[k] for k in joint_pos_keys], dtype=np.float32)

        images = {}
        for cam in CAMERA_KEYS:
            if cam not in raw:
                raise RuntimeError(f"Camera {cam!r} missing from robot observation: {list(raw.keys())}")
            img = raw[cam]  # HxWx3 BGR uint8 from RealSense
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[cam] = np.transpose(img, (2, 0, 1))  # CHW

        return {
            "state": joint_positions,
            "images": images,
            "prompt": self.task_prompt,
            "_raw_high": raw[CAMERA_KEYS[0]],  # carried out-of-band for frame saving
        }

    def _send_action(self, action: np.ndarray) -> None:
        joint_features = list(self.robot._joint_ft.keys())
        if len(action) != len(joint_features):
            raise RuntimeError(
                f"Action dim {len(action)} != robot joint count {len(joint_features)}"
            )
        if self.test_mode:
            logger.info(f"TEST MODE: would send action: {np.round(action, 4).tolist()}")
            return
        action_dict = {k: float(action[i]) for i, k in enumerate(joint_features)}
        self.robot.send_action(action_dict)

    def _move_to_start(self, goal: np.ndarray, duration: float) -> None:
        """PCHIP-interpolate from current pose to goal over `duration` seconds."""
        raw = self.robot.get_observation()
        joint_pos_keys = [k for k in raw.keys() if k.endswith(".pos")]
        current = np.array([raw[k] for k in joint_pos_keys], dtype=np.float32)
        interp = PchipInterpolator(np.array([0.0, duration]), np.array([current, goal]), axis=0)

        t_start = time.time()
        while True:
            t = time.time() - t_start
            if t >= duration:
                break
            self._send_action(interp(t))
            time.sleep(self.dt)

    # ----- policy inference -----

    def _request_chunk(self, obs_for_policy: dict) -> np.ndarray:
        # Strip our side-channel key before sending.
        send = {k: v for k, v in obs_for_policy.items() if not k.startswith("_")}

        # Save the exact dict sent to the server (the "server input") for this inference.
        infer_dir = self.output_dir / "server_inputs" / f"infer_{self._inference_count:04d}"
        infer_dir.mkdir(parents=True, exist_ok=True)
        # Joint state + prompt go in a single .npz; images saved both as .npy (raw CHW RGB the
        # server gets) and as .jpg (for human inspection — converted back to HWC BGR for cv2).
        np.savez(
            infer_dir / "state_prompt.npz",
            state=send["state"],
            prompt=np.asarray(send["prompt"]),
            episode_step=self.episode_step,
        )
        for cam, chw_rgb in send["images"].items():
            np.save(infer_dir / f"{cam}.npy", chw_rgb)
            hwc_bgr = cv2.cvtColor(np.transpose(chw_rgb, (1, 2, 0)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(infer_dir / f"{cam}.jpg"), hwc_bgr)

        t0 = time.monotonic()
        response = self.policy_client.infer(send)
        elapsed = (time.monotonic() - t0) * 1000.0
        self._inference_count += 1
        chunk = np.asarray(response["actions"], dtype=np.float32)
        logger.info(
            f"Inference #{self._inference_count - 1}: chunk shape={chunk.shape}  "
            f"client_total={elapsed:.0f}ms  saved to {infer_dir}"
        )
        # Persist the returned chunk alongside the inputs for full reproducibility.
        np.save(infer_dir / "action_chunk.npy", chunk)
        return chunk

    # ----- per-step bookkeeping -----

    def _log_step(self, obs: dict, action: np.ndarray) -> None:
        state = obs["state"]
        if not self._csv_header_written:
            header = (
                ["step", "timestamp"]
                + [f"state_{i}" for i in range(len(state))]
                + [f"action_{i}" for i in range(len(action))]
            )
            self._csv_writer.writerow(header)
            self._csv_header_written = True
        self._csv_writer.writerow(
            [self.episode_step, time.time(), *state.tolist(), *action.tolist()]
        )
        self._csv_file.flush()

        if self.save_frames_dir is not None:
            self.save_frames_dir.mkdir(parents=True, exist_ok=True)
            # Save the raw high cam (full-res, what the camera produced).
            if "_raw_high" in obs:
                cv2.imwrite(
                    str(self.save_frames_dir / f"step_{self.episode_step:05d}_raw_cam_high.jpg"),
                    obs["_raw_high"],
                )
            # Save the actual processed images that would be sent to the policy (224x224 RGB).
            for cam, chw_rgb in obs["images"].items():
                hwc_bgr = cv2.cvtColor(np.transpose(chw_rgb, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(self.save_frames_dir / f"step_{self.episode_step:05d}_proc_{cam}.jpg"),
                    hwc_bgr,
                )

    # ----- main loop -----

    def run(self) -> None:
        print(
            "\nControls:\n"
            "  n = execute next single action\n"
            "  c = continuous: execute CHUNK_SIZE actions then re-plan, repeat\n"
            "  r = re-plan now (discard current chunk)\n"
            "  q = quit\n"
        )
        print(f"Prompt: {self.task_prompt!r}")
        print(f"Chunk size: {self.chunk_size}")
        print("Press a key to begin. First action will smoothly interpolate from current pose.\n")

        is_first_step = True
        continuous_remaining = 0  # how many auto-steps left in current `c` batch

        with KeyReader() as keys:
            while True:
                # Decide what to do this iteration
                if continuous_remaining > 0:
                    cmd = "n"  # internal "step" command while in continuous mode
                    continuous_remaining -= 1
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
                    self.action_chunk = None
                    self.chunk_idx = 0
                    logger.info("Re-plan requested; current chunk discarded.")
                    continue

                if cmd == "c":
                    # Force a fresh inference, then queue CHUNK_SIZE steps.
                    self.action_chunk = None
                    self.chunk_idx = 0
                    continuous_remaining = self.chunk_size
                    logger.info(f"Continuous mode: queuing {self.chunk_size} steps.")
                    continue

                if cmd != "n":
                    print(f"  unknown key {cmd!r}; valid: n c r q")
                    continue

                # ---- execute one step ----
                step_start = time.perf_counter()
                obs = self._read_observation()

                if self.action_chunk is None or self.chunk_idx >= self.chunk_size:
                    self.action_chunk = self._request_chunk(obs)
                    self.chunk_idx = 0

                action = self.action_chunk[self.chunk_idx]
                self.chunk_idx += 1

                if is_first_step:
                    logger.info(
                        f"First step: smooth-moving to first predicted action over "
                        f"{self.startup_warm_duration:.1f}s..."
                    )
                    self._move_to_start(action, duration=self.startup_warm_duration)
                    is_first_step = False
                else:
                    self._send_action(action)

                self._log_step(obs, action)
                self.episode_step += 1

                step_ms = (time.perf_counter() - step_start) * 1000.0
                logger.info(
                    f"step {self.episode_step:4d}  chunk_idx={self.chunk_idx-1}  "
                    f"loop={step_ms:.0f}ms"
                )

                # If we're in continuous mode, hold the 30Hz cadence.
                if continuous_remaining > 0:
                    sleep_left = self.dt - (time.perf_counter() - step_start)
                    if sleep_left > 0:
                        time.sleep(sleep_left)

    def cleanup(self) -> None:
        logger.info("Cleaning up.")
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        try:
            self.robot.disconnect()
        except Exception as e:
            logger.warning(f"robot.disconnect raised: {e}")


def main():
    parser = argparse.ArgumentParser(description="Real-robot eval client for OpenPI policy server.")
    parser.add_argument("--policy-host", default=POLICY_HOST, help="Policy server host (overrides POLICY_HOST in file)")
    parser.add_argument("--policy-port", type=int, default=POLICY_PORT, help="Policy server port (overrides POLICY_PORT)")
    parser.add_argument("--task-prompt", default=TASK_PROMPT, help="Task prompt (overrides TASK_PROMPT)")
    parser.add_argument("--chunk-size", type=int, default=25,
                        help="How many actions to consume per inference (and per `c` batch).")
    parser.add_argument("--control-freq", type=int, default=30, help="Hz (only for `c` cadence).")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save cam_high frames to <output-dir>/frames/")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--startup-duration", type=float, default=5.0,
                        help="Seconds to smoothly interpolate to first predicted action.")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: do NOT move the robot. Capture obs, request actions from "
                             "the server, log + save what would have been sent, but skip "
                             "robot.send_action(). Implies --save-frames.")
    args = parser.parse_args()

    if args.test:
        args.save_frames = True  # always save frames in test mode

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    frames_dir = (output_dir / "frames") if (args.save_frames and output_dir is not None) else (
        pathlib.Path("eval_real_outputs/_pending/frames") if args.save_frames else None
    )

    bridge = TrossenEvalBridge(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        task_prompt=args.task_prompt,
        chunk_size=args.chunk_size,
        control_frequency=args.control_freq,
        save_frames_dir=frames_dir,
        output_dir=output_dir,
        startup_warm_duration=args.startup_duration,
        test_mode=args.test,
    )
    # Resolve frames_dir under the actual output_dir if we deferred above.
    if args.save_frames and frames_dir is not None and "_pending" in str(frames_dir):
        bridge.save_frames_dir = bridge.output_dir / "frames"

    def _sigint(_signum, _frame):
        logger.info("SIGINT received; shutting down.")
        bridge.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    try:
        bridge.run()
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
