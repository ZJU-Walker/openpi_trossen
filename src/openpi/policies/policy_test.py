import flax.nnx as nnx
import jax
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms as _transforms
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize
from openpi.training import config as _config


class _EchoPrefixModel(nnx.Module):
    def __init__(self):
        self.action_horizon = 4
        self.action_dim = 4
        self.max_token_len = 8

    def sample_actions(self, rng, observation, *, action_prefix=None, prefix_length=None):
        del rng, observation, prefix_length
        return action_prefix


def test_action_prefix_transform_matches_action_target_path():
    action_prefix = np.array([[11.0, 18.0, 33.0], [12.0, 19.0, 34.0]], dtype=np.float32)
    state = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    norm_stats = {
        "actions": normalize.NormStats(
            mean=np.array([1.0, 10.0, 1.0], dtype=np.float32),
            std=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
    }
    transforms = [
        _transforms.DeltaActions(mask=[True, False, True]),
        _transforms.Normalize(norm_stats),
        _transforms.PadStatesAndActions(model_action_dim=4),
    ]
    obs = {
        "image": {
            "base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "state": state,
        "action_prefix": action_prefix,
        "prefix_length": 2,
    }
    expected = _transforms.compose(transforms)(
        {
            "image": obs["image"],
            "image_mask": obs["image_mask"],
            "state": state.copy(),
            "actions": action_prefix.copy(),
        }
    )["actions"]
    expected = np.pad(expected, ((0, 2), (0, 0)))

    policy = _policy.Policy(_EchoPrefixModel(), rng=jax.random.key(0), transforms=transforms)
    result = policy.infer(obs)

    np.testing.assert_allclose(result["actions"], expected)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
