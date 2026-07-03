"""Robot-workstation client: stream cam_high to the Qwen server and measure realtime speed.

Runs on the robot computer (has `openpi_client` + the lerobot fork). Captures cam_high frames, streams
them to the H200 server, and sweeps {cached, full} x {8,16,36} early-exit depths, reporting end-to-end
round-trip latency vs server inference time vs network overhead. Writes realtime/results_realtime.md.

Usage (robot workstation):
    python realtime/client.py --policy_host <H200-host> --port 8000 --steps 50

Mirrors the capture pattern of examples/trossen_ai/eval_real_hierarchical.py (RobotSource): cam_high is
captured HWC uint8 and resized to the model resolution before sending (small payload).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque

import numpy as np

LEROBOT_FORK_PATH = "/home/iris/lerobot"
MODES = ["cached", "full"]
LAYERS = [8, 16, 36]
WINDOW_FRAMES = 30
RESULTS_PATH = "realtime/results_realtime.md"


# --------------------------------------------------------------------------- #
class RobotCamera:
    """cam_high capture via the lerobot Trossen fork (lazy import)."""

    def __init__(self, send_res: int):
        if LEROBOT_FORK_PATH not in sys.path:
            sys.path.insert(0, LEROBOT_FORK_PATH)
        from lerobot.common.robot_devices.robots.configs import TrossenAISoloRobotConfig
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        cfg = TrossenAISoloRobotConfig(
            max_relative_target=None, min_time_to_move_multiplier=4.0, camera_interface="opencv"
        )
        self.robot = make_robot_from_config(cfg)
        self.robot.connect()
        self.send_res = send_res
        import cv2  # noqa
        self._cv2 = cv2

    def capture(self):
        """Return (frame HWC uint8 [send_res,send_res,3], state float32 [14])."""
        obs = self.robot.capture_observation()
        state = obs["observation.state"].detach().cpu().numpy().astype(np.float32)
        img = obs["observation.images.cam_high"].detach().cpu().numpy()  # HWC uint8
        img = self._cv2.resize(img, (self.send_res, self.send_res))
        return img.astype(np.uint8), state

    def close(self):
        self.robot.disconnect()


# --------------------------------------------------------------------------- #
def run_phase(client, cam, mode, layers, steps, warmup, history):
    """One sweep cell. cached: send 2 newest frames; full: send the 30-frame window."""
    n_send = 2 if mode == "cached" else WINDOW_FRAMES
    client.infer({"mode": "reset"})

    def grab_obs():
        img, state = cam.capture()
        history.append(img)
        while len(history) < n_send:           # bootstrap: duplicate until enough history
            history.append(img)
        frames = np.stack(list(history)[-n_send:], axis=0)
        return {"mode": mode, "num_layers": layers, "frames": frames, "state": state}

    for _ in range(warmup):
        client.infer(grab_obs())

    rtt, server_ms, vis_ms, lm_ms, payload = [], [], [], [], 0
    for _ in range(steps):
        obs = grab_obs()
        payload = len(client._packer.pack(obs)) if hasattr(client, "_packer") else 0
        t0 = time.perf_counter()
        resp = client.infer(obs)
        rtt.append((time.perf_counter() - t0) * 1000.0)
        server_ms.append(resp["server_timing"]["infer_ms"])
        vis_ms.append(resp.get("vision_ms", 0.0))
        lm_ms.append(resp.get("lm_ms", 0.0))

    return {
        "mode": mode, "layers": layers,
        "rtt_med": statistics.median(rtt), "rtt_p95": float(np.percentile(rtt, 95)),
        "server_ms": statistics.mean(server_ms),
        "vision_ms": statistics.mean(vis_ms), "lm_ms": statistics.mean(lm_ms),
        "net_ms": statistics.median(rtt) - statistics.mean(server_ms),
        "payload_KB": payload / 1024.0, "hz": 1000.0 / statistics.median(rtt),
    }


def write_results(rows, meta):
    lines = [
        "# Realtime client/server speed test (robot cam_high -> H200 Qwen server)",
        "",
        f"Server: {meta}",
        "",
        "Per-step end-to-end: `rtt` = client round-trip (send->infer->recv); `server` = server-side",
        "infer wall time; `net` = rtt - server (network + serialization); `payload` = request bytes.",
        "",
        "| mode | layers | rtt_med_ms | rtt_p95_ms | server_ms | vision_ms | lm_ms | net_ms | payload_KB | Hz |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['layers']} | {r['rtt_med']:.1f} | {r['rtt_p95']:.1f} | "
            f"{r['server_ms']:.1f} | {r['vision_ms']:.1f} | {r['lm_ms']:.1f} | {r['net_ms']:.1f} | "
            f"{r['payload_KB']:.0f} | {r['hz']:.1f} |"
        )
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS_PATH}")


def run_live(client, cam, mode, layers, send_res, no_viz=False):
    """Continuous loop: capture -> send -> show the sent frame + print latency every step. Press 'q' to quit."""
    cv2 = None
    if not no_viz:
        import cv2  # only needed for the window

    n_send = 2 if mode == "cached" else WINDOW_FRAMES
    history = deque(maxlen=WINDOW_FRAMES)
    win = f"qwen sent frame [{mode} L{layers}]"
    client.infer({"mode": "reset"})
    print(f"LIVE {mode} L{layers}: streaming cam_high to the server. Press 'q' in the window (or Ctrl-C) to stop.")

    step = 0
    try:
        while True:
            img, state = cam.capture()          # img: RGB [send_res,send_res,3]
            history.append(img)
            while len(history) < n_send:
                history.append(img)
            frames = np.stack(list(history)[-n_send:], axis=0)

            obs = {"mode": mode, "num_layers": layers, "frames": frames, "state": state}
            payload_kb = len(client._packer.pack(obs)) / 1024.0
            t0 = time.perf_counter()
            resp = client.infer(obs)
            rtt = (time.perf_counter() - t0) * 1000.0

            srv = resp["server_timing"]["infer_ms"]
            vis, lm = resp.get("vision_ms", 0.0), resp.get("lm_ms", 0.0)
            ready = resp.get("ready", True)
            tag = "READY" if ready else f"warmup {resp.get('num_pairs', '?')}/15"
            print(f"step {step:4d}  rtt={rtt:6.1f}ms  server={srv:6.1f}ms (vis={vis:5.1f} lm={lm:5.1f})  "
                  f"net={rtt - srv:5.1f}ms  payload={payload_kb:.0f}KB  {1000.0 / rtt:5.1f}Hz  [{tag}]")

            if not no_viz:
                disp = cv2.cvtColor(history[-1], cv2.COLOR_RGB2BGR)   # newest sent frame
                disp = cv2.resize(disp, (512, 512), interpolation=cv2.INTER_NEAREST)
                lines = [f"{mode}  L{layers}  [{tag}]",
                         f"rtt {rtt:.0f}ms  server {srv:.0f}ms",
                         f"vis {vis:.0f}  lm {lm:.0f}  net {rtt - srv:.0f}ms"]
                y = 24
                for txt in lines:
                    cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    y += 26
                cv2.imshow(win, disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            step += 1
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if not no_viz:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_host", required=True, help="H200 server host")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", default="cached", choices=["cached", "full", "sweep"],
                    help="cached/full = live loop (window + per-step latency); sweep = timed table over modes x layers")
    ap.add_argument("--layers", type=int, default=16, help="early-exit depth for live mode")
    ap.add_argument("--no_viz", action="store_true", help="live mode: print latency only, no window")
    ap.add_argument("--steps", type=int, default=50, help="sweep mode: timed steps per cell")
    ap.add_argument("--warmup", type=int, default=18, help=">=15 fills the cached pair-cache")
    ap.add_argument("--send_res", type=int, default=336)
    args = ap.parse_args()

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.port)
    meta = getattr(client, "_server_metadata", {})
    print("connected; server metadata:", meta)

    cam = RobotCamera(args.send_res)

    if args.mode != "sweep":
        try:
            run_live(client, cam, args.mode, args.layers, args.send_res, no_viz=args.no_viz)
        finally:
            cam.close()
        return

    history = deque(maxlen=WINDOW_FRAMES)
    rows = []
    try:
        for mode in MODES:
            for layers in LAYERS:
                history.clear()
                r = run_phase(client, cam, mode, layers, args.steps, args.warmup, history)
                print(f"  {mode:>6} L{layers:<2}: rtt={r['rtt_med']:.1f}ms server={r['server_ms']:.1f} "
                      f"(vis={r['vision_ms']:.1f} lm={r['lm_ms']:.1f}) net={r['net_ms']:.1f} "
                      f"payload={r['payload_KB']:.0f}KB {r['hz']:.1f}Hz")
                rows.append(r)
    finally:
        cam.close()

    write_results(rows, meta)


if __name__ == "__main__":
    main()