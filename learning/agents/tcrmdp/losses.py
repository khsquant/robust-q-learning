"""Losses for TCRMDP-style TD3 algorithms."""

from typing import Any

import jax
import jax.numpy as jnp

from brax.training.types import Params, PRNGKey
from agents.tcrmdp import common
from agents.tcrmdp import networks as tcrmdp_networks


def _obs(key: str, value: jax.Array) -> dict[str, jax.Array]:
  return {key: value}


def make_losses(
    tcrmdp_network: tcrmdp_networks.TCRMDPNetworks,
    reward_scaling: float,
    discounting: float,
    dr_low: jax.Array,
    dr_high: jax.Array,
    radius: float,
    omniscient_adversary: bool = True,
    asymmetric_critic: bool = True,
):
  """Creates critic and actor losses."""
  agent_policy = tcrmdp_network.agent_policy_network
  agent_q = tcrmdp_network.agent_q_network
  adversary_policy = tcrmdp_network.adversary_policy_network
  adversary_q = tcrmdp_network.adversary_q_network
  algorithm = tcrmdp_network.algorithm

  def agent_critic_loss(
      q_params: Params,
      target_policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions,
      noise: jax.Array,
      key: PRNGKey,
  ):
    del key
    q_old_action = agent_q.apply(
        normalizer_params,
        q_params,
        _obs("critic_state", transitions.critic_observation),
        transitions.action,
    )
    next_action = agent_policy.apply(
        normalizer_params,
        target_policy_params,
        _obs("actor_state", transitions.next_actor_observation),
    )
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    next_q = agent_q.apply(
        normalizer_params,
        target_q_params,
        _obs("critic_state", transitions.next_critic_observation),
        next_action,
    )
    next_v = jnp.min(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)
    truncation = transitions.extras["state_extras"]["truncation"]
    q_error *= jnp.expand_dims(1 - truncation, -1)
    return 0.5 * jnp.mean(jnp.square(q_error)), (q_old_action, next_v)

  def agent_actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions,
      key: PRNGKey,
  ):
    del key
    action = agent_policy.apply(
        normalizer_params,
        policy_params,
        _obs("actor_state", transitions.actor_observation),
    )
    q_action = agent_q.apply(
        normalizer_params,
        q_params,
        _obs("critic_state", transitions.critic_observation),
        action,
    )
    return -jnp.mean(jnp.min(q_action, axis=-1))

  def rarl_adversary_critic_loss(
      q_params: Params,
      target_policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions,
      noise: jax.Array,
      key: PRNGKey,
  ):
    del key
    q_old_action = adversary_q.apply(
        normalizer_params,
        q_params,
        _obs("adv_state", transitions.adv_observation),
        transitions.adv_action,
    )
    next_adv_action = adversary_policy.apply(
        normalizer_params,
        target_policy_params,
        _obs("adv_state", transitions.next_adv_observation),
    )
    next_adv_action = jnp.clip(next_adv_action + noise, -1.0, 1.0)
    next_q = adversary_q.apply(
        normalizer_params,
        target_q_params,
        _obs("adv_state", transitions.next_adv_observation),
        next_adv_action,
    )
    next_v = jnp.min(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(
        transitions.adv_reward * reward_scaling
        + transitions.discount * discounting * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)
    truncation = transitions.extras["state_extras"]["truncation"]
    q_error *= jnp.expand_dims(1 - truncation, -1)
    return 0.5 * jnp.mean(jnp.square(q_error)), (q_old_action, next_v)

  def rarl_adversary_actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions,
      key: PRNGKey,
  ):
    del key
    adv_action = adversary_policy.apply(
        normalizer_params,
        policy_params,
        _obs("adv_state", transitions.adv_observation),
    )
    q_action = adversary_q.apply(
        normalizer_params,
        q_params,
        _obs("adv_state", transitions.adv_observation),
        adv_action,
    )
    return -jnp.mean(jnp.min(q_action, axis=-1))

  def m2td3_adversary_actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions,
      key: PRNGKey,
  ):
    del key
    adv_action = adversary_policy.apply(
        normalizer_params,
        policy_params,
        _obs("adv_state", transitions.adv_observation),
    )
    proposed_params = common.update_params(
        transitions.dynamics_params, adv_action, dr_low, dr_high, radius
    )
    _, proposed_critic_obs, _ = common.build_observations(
        transitions.raw_observation,
        proposed_params,
        transitions.action,
        dr_low,
        dr_high,
        algorithm,
        omniscient_adversary,
        asymmetric_critic,
    )
    q_action = agent_q.apply(
        normalizer_params,
        q_params,
        _obs("critic_state", proposed_critic_obs),
        transitions.action,
    )
    return jnp.mean(jnp.min(q_action, axis=-1))

  return (
      agent_critic_loss,
      agent_actor_loss,
      rarl_adversary_critic_loss,
      rarl_adversary_actor_loss,
      m2td3_adversary_actor_loss,
  )
