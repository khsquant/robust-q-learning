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

"""TD3 losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any, Sequence, Tuple

from brax.training import types
from agents.gmmtd3 import networks as gmmtd3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp
from brax import envs

from learning.agents.gmmtd3.train import TransitionwithGMMParams
from learning.module.gmmvi.network import GMMTrainingState
_PMAP_AXIS_NAME = 'i'
from brax.training.acme import running_statistics
from brax.envs.base import Wrapper, Env, State

def make_losses(
    gmmtd3_network: gmmtd3_networks.GMMTd3Networks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    dr_augmented_critic: bool = False,
):
  """Creates the td3 losses."""

  target_entropy = -0.5 * action_size
  policy_network = gmmtd3_network.policy_network
  q_network = gmmtd3_network.q_network
  gmm_network = gmmtd3_network.gmm_network

  def _q_apply(normalizer_params, q_params, observation, action, dynamics_params):
    if dr_augmented_critic:
      return q_network.apply(
          normalizer_params, q_params, observation, action, dynamics_params
      )
    return q_network.apply(normalizer_params, q_params, observation, action)

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions: Any,
      noise: jnp.ndarray,
      key: PRNGKey,
  ) -> jnp.ndarray:
    q_old_action = _q_apply(
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
    next_q = _q_apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
        transitions.dynamics_params,
    )
    next_v = jnp.min(next_q, axis=-1) 
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss, (q_old_action, next_v)

  def actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions: Any,
      key: PRNGKey,
  ) -> jnp.ndarray:
    action = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    q_action = _q_apply(
        normalizer_params,
        q_params,
        transitions.observation,
        action,
        transitions.dynamics_params,
    )
    min_q = jnp.min(q_action, axis=-1)
    return -jnp.mean(min_q)

    
  def gmm_update(
        gmmvi_state, key
  ):
    samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads = \
        gmm_network.sample_selector.select_train_datas(gmmvi_state.sample_db_state)
    new_component_stepsizes = gmm_network.component_stepsize_fn(gmmvi_state.model_state)
    new_model_state = gmm_network.model.update_stepsizes(gmmvi_state.model_state, new_component_stepsizes)
    expected_hessian_neg, expected_grad_neg = gmm_network.more_ng_estimator(new_model_state,
                                                            samples,
                                                            sample_dist_densities,
                                                            target_lnpdfs,
                                                            target_lnpdf_grads)
    new_model_state = gmm_network.component_updater(new_model_state,
                                    expected_hessian_neg,
                                    expected_grad_neg,
                                    new_model_state.stepsizes)
        
    # new_weight_stepsize_adapter_state = weight_stepsize_adapter.update_stepsize(train_state.weight_stepsize_adapter_state, new_model_state)
    new_model_state = gmm_network.weight_updater(new_model_state, samples, sample_dist_densities, target_lnpdfs,
                                                    gmmvi_state.weight_stepsize)
    new_num_updates = gmmvi_state.num_updates + 1
    new_model_state, new_component_adapter_state, new_sample_db_state = \
        gmm_network.component_adapter(gmmvi_state.component_adaptation_state,
                                                    gmmvi_state.sample_db_state,
                                                    new_model_state,
                                                    new_num_updates,
                                                    key)

    return GMMTrainingState(temperature=gmmvi_state.temperature,
                        model_state=new_model_state,
                        component_adaptation_state=new_component_adapter_state,
                        num_updates=new_num_updates,
                        sample_db_state=new_sample_db_state,
                        weight_stepsize=gmmvi_state.weight_stepsize)

  return critic_loss, actor_loss, gmm_update
