import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0 as _pi0
from openpi.models import pi0_config


def _tiny_config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        siglip_variant="mu/224",
        dtype="float32",
        action_dim=4,
        action_horizon=6,
        max_token_len=8,
        **kwargs,
    )


def test_posemb_sincos_accepts_per_token_timesteps():
    timestep = jnp.zeros((2, 6), dtype=jnp.float32)

    embedding = _pi0.posemb_sincos(timestep, embedding_dim=8, min_period=4e-3, max_period=4.0)

    assert embedding.shape == (2, 6, 8)


def test_embed_suffix_accepts_per_token_timesteps_and_uses_clean_prefix_value():
    config = _tiny_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    actions = config.fake_act(batch_size=2)
    prefix_lengths = jnp.array([2, 4], dtype=jnp.int32)
    prefix_mask = _pi0._rtc_prefix_mask(prefix_lengths, config.action_horizon)  # noqa: SLF001
    timestep = jnp.where(prefix_mask, 0.0, 0.5)

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, actions, timestep)

    assert suffix_tokens.shape[:2] == (2, config.action_horizon)
    assert suffix_mask.shape == (2, config.action_horizon)
    assert suffix_ar_mask.shape == (config.action_horizon,)
    assert adarms_cond is not None
    assert adarms_cond.shape == (2, config.action_horizon, 64)
    np.testing.assert_array_equal(np.asarray(timestep[prefix_mask]), np.zeros(6, dtype=np.float32))


def test_rtc_loss_mask_zeroes_prefix_and_reweights_postfix():
    loss = jnp.ones((2, 6), dtype=jnp.float32)
    prefix_lengths = jnp.array([0, 3], dtype=jnp.int32)

    masked = _pi0._apply_rtc_loss_mask(loss, prefix_lengths)  # noqa: SLF001

    np.testing.assert_allclose(np.asarray(masked[0]), np.ones(6, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(masked[1, :3]), np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(masked[1, 3:]), np.full(3, 2.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(masked.mean(axis=-1)), np.ones(2, dtype=np.float32))


def test_compute_loss_masks_committed_prefix_tokens():
    config = _tiny_config(rtc_prefix_min_length=2, rtc_prefix_max_length=2)
    model = config.create(jax.random.key(0))

    loss = model.compute_loss(jax.random.key(1), config.fake_obs(batch_size=2), config.fake_act(batch_size=2))

    assert loss.shape == (2, config.action_horizon)
    np.testing.assert_allclose(np.asarray(loss[:, :2]), np.zeros((2, 2), dtype=np.float32), atol=1e-6)
    assert np.all(np.isfinite(np.asarray(loss[:, 2:])))


def test_sample_actions_returns_action_prefix_exactly():
    config = _tiny_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    noise = jnp.zeros((2, config.action_horizon, config.action_dim), dtype=jnp.float32)
    action_prefix = jnp.arange(2 * config.action_horizon * config.action_dim, dtype=jnp.float32).reshape(
        2, config.action_horizon, config.action_dim
    )
    prefix_lengths = jnp.array([2, 4], dtype=jnp.int32)

    actions = model.sample_actions(
        jax.random.key(1),
        obs,
        num_steps=2,
        noise=noise,
        action_prefix=action_prefix,
        prefix_length=prefix_lengths,
    )

    np.testing.assert_allclose(np.asarray(actions[0, :2]), np.asarray(action_prefix[0, :2]), atol=0.0)
    np.testing.assert_allclose(np.asarray(actions[1, :4]), np.asarray(action_prefix[1, :4]), atol=0.0)
