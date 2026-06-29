# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BridgeTD3 losses."""

from typing import Any

import jax
import jax.numpy as jnp

from agents.bridgetd3 import networks as bridgetd3_networks
from agents.tcrmdp import common as tc_common
from brax.training.types import Params, PRNGKey


def make_losses(
    bridgetd3_network: bridgetd3_networks.BridgeTd3Networks,
    reward_scaling: float,
    discounting: float,
    dr_low: jax.Array,
    dr_high: jax.Array,
    target_kinetic: float,
    use_tc: bool = False,
    radius: float = 0.001,
):
  """Creates critic, actor, and bridge-adversary losses."""
  policy_network = bridgetd3_network.policy_network
  q_network = bridgetd3_network.q_network
  adversary_network = bridgetd3_network.adversary_network

  def alpha_loss(
      log_alpha: jax.Array,
      kinetic_mean: jax.Array,
  ):
    return log_alpha * jax.lax.stop_gradient(target_kinetic - kinetic_mean)

  def effective_params(
      current_params: jax.Array,
      candidate_params: jax.Array,
  ) -> jax.Array:
    if not use_tc:
      return candidate_params
    return tc_common.clip_params_to_radius(
        current_params,
        candidate_params,
        dr_low,
        dr_high,
        radius,
    )

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      adversary_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions,
      noise: jnp.ndarray,
      key: PRNGKey,
  ):
    q_old_action = q_network.apply(
        normalizer_params,
        q_params,
        transitions.observation,
        transitions.action,
        transitions.dynamics_params,
    )
    next_action = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    candidate_next_params, _, _ = adversary_network.apply(
        normalizer_params,
        adversary_params,
        transitions.next_observation,
        key,
        dr_low,
        dr_high,
        initial_params=transitions.dynamics_params if use_tc else None,
        tc_radius=radius if use_tc else None,
        deterministic=False,
    )
    next_params = effective_params(
        transitions.dynamics_params,
        candidate_next_params,
    )
    next_q = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
        next_params,
    )
    next_v = jnp.min(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)
    truncation = transitions.extras["state_extras"]["truncation"]
    q_error *= jnp.expand_dims(1 - truncation, -1)
    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss, (q_old_action, next_v)

  def actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      adversary_params: Params,
      transitions,
      key: PRNGKey,
  ):
    action = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    candidate_params, _, _ = adversary_network.apply(
        normalizer_params,
        adversary_params,
        transitions.observation,
        key,
        dr_low,
        dr_high,
        initial_params=transitions.dynamics_params if use_tc else None,
        tc_radius=radius if use_tc else None,
        deterministic=False,
    )
    dynamics_params = effective_params(
        transitions.dynamics_params,
        candidate_params,
    )
    q_action = q_network.apply(
        normalizer_params,
        q_params,
        transitions.observation,
        action,
        dynamics_params,
    )
    return -jnp.mean(jnp.min(q_action, axis=-1))

  def adversary_loss(
      adversary_params: Params,
      normalizer_params: Any,
      policy_params: Params,
      q_params: Params,
      alpha: jax.Array,
      transitions,
      key: PRNGKey,
  ):
    action = jax.lax.stop_gradient(
        policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
    )
    candidate_params, kinetic, _ = adversary_network.apply(
        normalizer_params,
        adversary_params,
        transitions.observation,
        key,
        dr_low,
        dr_high,
        initial_params=transitions.dynamics_params if use_tc else None,
        tc_radius=radius if use_tc else None,
        deterministic=False,
    )
    dynamics_params = effective_params(
        transitions.dynamics_params,
        candidate_params,
    )
    q_action = q_network.apply(
        normalizer_params,
        q_params,
        transitions.observation,
        action,
        dynamics_params,
    )
    worst_q = jnp.min(q_action, axis=-1)
    loss = jnp.mean(alpha * kinetic + worst_q)
    aux = {
        "adversary_kinetic_mean": kinetic.mean(),
        "adversary_q_mean": worst_q.mean(),
        "adversary_params_mean": dynamics_params.mean(),
        "adversary_params_std": dynamics_params.std(),
    }
    return loss, aux

  return critic_loss, actor_loss, adversary_loss, alpha_loss
