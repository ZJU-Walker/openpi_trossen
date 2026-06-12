#!/usr/bin/env python3
"""Mock Qwen subtask server for testing the hierarchical driver WITHOUT a GPU.

Speaks the same websocket protocol as serve_subtask_predictor.py but replays
subtask predictions from an offline stage-1 JSON (predict_subtasks_lerobot.py
output), keyed by the newest frame index in the request, with an optional
artificial latency to mimic the real model.

Useful for: robot-workstation integration tests, driver development, and
single-GPU machines where the real Qwen does not fit beside pi0.5.

Usage:
    python examples/trossen_ai/mock_subtask_server.py \
        --preds-json <stage1>/subtask_preds_ep000027.json --port 8001
"""

import argparse
import asyncio
import http
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)

PROMPT_MAP = {
    "observe_human": "wait and observe the human",
    "put_green_block_to_plate": "put the green block to the plate",
    "put_yellow_block_to_plate": "put the yellow block to the plate",
}


class MockService:
    def __init__(self, preds_json: Path, latency_ms: float):
        data = json.loads(preds_json.read_text())
        self._preds = sorted((int(r["frame"]), r["pred"]) for r in data["frames"])
        self._latency_s = latency_ms / 1000.0
        logger.info(f"Loaded {len(self._preds)} predictions from {preds_json} "
                    f"(episode {data['episode']})")

    def infer(self, obs: dict) -> dict:
        t = int(np.asarray(obs["frame_indices"]).reshape(-1)[-1])
        pred = self._preds[0][1]
        for frame, p in self._preds:
            if frame > t:
                break
            pred = p
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        return {"subtask": pred, "lowlevel_prompt": PROMPT_MAP.get(pred),
                "valid": pred in PROMPT_MAP}


def _handler_factory(service, metadata):
    async def handler(websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                start = time.monotonic()
                result = service.infer(obs)
                result["server_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.")
                raise

    return handler


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds-json", type=Path, required=True)
    p.add_argument("--latency-ms", type=float, default=300.0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()

    service = MockService(args.preds_json, args.latency_ms)
    metadata = {
        "service": "mock_subtask_predictor",
        "preds_json": str(args.preds_json),
        "labels": sorted(PROMPT_MAP),
        "prompt_map": PROMPT_MAP,
        "history_seconds": 5.0,
        "num_frames": 5,
        "fps": 30.0,
    }

    async def serve():
        handler = _handler_factory(service, metadata)
        async with _server.serve(
            handler, args.host, args.port,
            compression=None, max_size=None, ping_interval=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"Mock subtask server on ws://{args.host}:{args.port}")
            await server.serve_forever()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
