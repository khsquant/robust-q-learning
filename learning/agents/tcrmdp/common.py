"""Shared utilities for TCRMDP-style parameter control."""

from typing import Mapping, Tuple

import jax
import jax.numpy as jnp

from brax.training import types
from agents.tcrmdp import networks as tcrmdp_networks


def raw_state(obs: types.Observation) -> jax.Array:
  if isinstance(obs, Mapping):
    return obs["state"]
  return obs


def normalize_params(
    params: jax.Array,
    dr_low: jax.Array,
    dr_high: jax.Array,
) -> jax.Array:
  denom = jnp.maximum(dr_high - dr_low, 1e-6)
  return (params - dr_low) / denom


def update_params(
    params: jax.Array,
    adv_action: jax.Array,
    dr_low: jax.Array,
    dr_high: jax.Array,
    radius: float,
) -> jax.Array:
  return jnp.clip(params + radius * adv_action, dr_low, dr_high)


def clip_params_to_radius(
    params: jax.Array,
    candidate_params: jax.Array,
    dr_low: jax.Array,
    dr_high: jax.Array,
    radius: float,
) -> jax.Array:
  local_low = jnp.maximum(dr_low, params - radius)
  local_high = jnp.minimum(dr_high, params + radius)
  return jnp.clip(candidate_params, local_low, local_high)


def action_to_params(
    adv_action: jax.Array,
    dr_low: jax.Array,
    dr_high: jax.Array,
    scale: float = 1.0,
) -> jax.Array:
  dr_mid = (dr_low + dr_high) / 2.0
  dr_scale = (dr_high - dr_low) / 2.0
  return jnp.clip(dr_mid + scale * dr_scale * adv_action, dr_low, dr_high)


def build_observations(
    obs: types.Observation,
    params: jax.Array,
    agent_action: jax.Array,
    dr_low: jax.Array,
    dr_high: jax.Array,
    algorithm: str,
    omniscient_adversary: bool = True,
    asymmetric_critic: bool = True,
    dr_augmented_critic: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  del dr_low, dr_high
  state = raw_state(obs)
  psi = params
  state_and_psi = jnp.concatenate([state, psi], axis=-1)
  if algorithm == tcrmdp_networks.RARL:
    actor_obs = state
    critic_obs = state
    adv_obs = state
    if omniscient_adversary:
      adv_obs = jnp.concatenate([adv_obs, agent_action], axis=-1)
    return actor_obs, critic_obs, adv_obs
  actor_obs = state
  if dr_augmented_critic:
    critic_obs = state
  else:
    critic_obs = state_and_psi if asymmetric_critic else actor_obs
  adv_obs = jnp.concatenate([state_and_psi, agent_action], axis=-1)
  return actor_obs, critic_obs, adv_obs


def observation_dict(
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    adv_obs: jax.Array,
) -> dict[str, jax.Array]:
  return {
      "actor_state": actor_obs,
      "critic_state": critic_obs,
      "adv_state": adv_obs,
  }
