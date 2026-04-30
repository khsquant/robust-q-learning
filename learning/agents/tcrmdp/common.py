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


def action_to_params(
    adv_action: jax.Array,
    dr_low: jax.Array,
    dr_high: jax.Array,
) -> jax.Array:
  dr_mid = (dr_low + dr_high) / 2.0
  dr_scale = (dr_high - dr_low) / 2.0
  return jnp.clip(dr_mid + dr_scale * adv_action, dr_low, dr_high)


def build_observations(
    obs: types.Observation,
    params: jax.Array,
    agent_action: jax.Array,
    dr_low: jax.Array,
    dr_high: jax.Array,
    algorithm: str,
    omniscient_adversary: bool = True,
    asymmetric_critic: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  state = raw_state(obs)
  psi = normalize_params(params, dr_low, dr_high)
  state_and_psi = jnp.concatenate([state, psi], axis=-1)
  if algorithm == tcrmdp_networks.RARL:
    actor_obs = state
    critic_obs = state
    adv_obs = state
    if omniscient_adversary:
      adv_obs = jnp.concatenate([adv_obs, agent_action], axis=-1)
    return actor_obs, critic_obs, adv_obs
  if algorithm == tcrmdp_networks.VANILLA_TC_M2TD3:
    actor_obs = state
  else:
    actor_obs = state_and_psi
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
