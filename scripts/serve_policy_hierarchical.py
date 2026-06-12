"""Serve a subtask-conditioned pi0.5 checkpoint for HIERARCHICAL deployment.

Same as scripts/serve_policy.py policy:checkpoint, with one difference: the
subtask training configs inject per-frame training prompts via
PromptFromSubtaskSegments, which requires dataset episode/frame indices and
would crash (or override the prompt) on live observations. At deployment the
prompt comes from the high-level policy through obs["prompt"], so this wrapper
drops that transform before building the policy.

Usage:
    uv run scripts/serve_policy_hierarchical.py \
        --config=pi05_trossen_block_mem_subtask_full_delta_rtc \
        --dir=checkpoints/pi05_trossen_block_mem_subtask_full_delta_rtc/<exp>/<step> \
        --port=8000
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str
    """Training config name (e.g. pi05_trossen_block_mem_subtask_full_delta_rtc)."""

    dir: str
    """Checkpoint step directory (e.g. checkpoints/<config>/<exp>/9000)."""

    port: int = 8000
    default_prompt: str | None = "wait and observe the human"
    """Used only if a request arrives without obs["prompt"]."""


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    if getattr(train_config.data, "prompt_input_transform", None) is not None:
        logging.info("Dropping prompt_input_transform for live serving "
                     "(prompt comes from the high-level policy via obs['prompt']).")
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, prompt_input_transform=None),
        )

    policy = _policy_config.create_trained_policy(
        train_config, args.dir, default_prompt=args.default_prompt)

    hostname = socket.gethostname()
    logging.info("Serving %s from %s on %s:%d", args.config, args.dir, hostname, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=train_config.policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
