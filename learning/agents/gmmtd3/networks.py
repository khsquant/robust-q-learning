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

"""gmmtd3 networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from learning.module.gmmvi.network import create_gmm_network_and_state
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import jax

@flax.struct.dataclass
class GMMTd3Networks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  gmm_network: networks.FeedForwardNetwork

def make_inference_fn(gmmtd3_networks: GMMTd3Networks):
  """Creates params and inference function for the td3 agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False,
  ) -> types.Policy:

    def deterministic_policy(
        observations: types.Observation,
        key: PRNGKey = None,
    ) -> Tuple[types.Action, types.Extra]:
      return gmmtd3_networks.policy_network.apply(*params, observations), None

    def stochastic_policy(
        observations: types.Observation,
        noise_scales: jnp.ndarray,
        key: PRNGKey,
    ):  
        act = gmmtd3_networks.policy_network.apply(*params, observations)
        noise = jax.random.normal(key, shape=act.shape) * noise_scales[..., None]
        return act + noise, None
    if deterministic:
        return deterministic_policy
    else:
        return stochastic_policy

  return make_policy
def make_gmmtd3_networks(
    observation_size: int,
    action_size: int,
    dynamics_param_size : int,
    num_envs :int, 
    batch_size : int,
    init_key :jax.random.PRNGKey,
    param_size: int = 0,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    bound_info : Tuple = None,
    distributional_q : bool= False,
    num_atoms: int = 101,
    v_min: float = 0.,
    v_max: float = 0.,
    dr_augmented_critic: bool = False,
) -> GMMTd3Networks:
  """Make td3 networks."""

  policy_network = networks.make_deterministic_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key = policy_obs_key
  )
  del distributional_q, num_atoms, v_min, v_max
  if dr_augmented_critic:
    q_network = networks.make_augmented_q_network(
        observation_size,
        action_size,
        param_size or dynamics_param_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        obs_key=value_obs_key,
    )
  else:
    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        obs_key=value_obs_key,
    )
  init_gmmvi_state, gmm_network = create_gmm_network_and_state(dynamics_param_size, num_envs, batch_size, init_key,\
                                                                bound_info=bound_info)
  return GMMTd3Networks(
      policy_network=policy_network,
      q_network=q_network,
      gmm_network=gmm_network,
  ), init_gmmvi_state
