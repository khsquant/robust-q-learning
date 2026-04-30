# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Networks for TCRMDP-style TD3 algorithms."""

from typing import Mapping, Sequence, Tuple

import flax
from flax import linen
import jax
import jax.numpy as jnp

from brax.training import types
from brax.training.types import PRNGKey
from module import networks


VANILLA_TC_M2TD3 = "vanilla_tc_m2td3"
RARL = "rarl"
TC_RARL = "tc_rarl"
TC_M2TD3 = "tc_m2td3"


@flax.struct.dataclass
class TCRMDPNetworks:
  agent_policy_network: networks.FeedForwardNetwork
  agent_q_network: networks.FeedForwardNetwork
  adversary_policy_network: networks.FeedForwardNetwork
  adversary_q_network: networks.FeedForwardNetwork
  dynamics_param_size: int
  algorithm: str


def _raw_state(observations: types.Observation):
  if isinstance(observations, Mapping):
    if "actor_state" in observations:
      return observations["actor_state"]
    return observations["state"]
  return observations


def make_inference_fn(tcrmdp_networks: TCRMDPNetworks):
  """Creates params and inference function for the agent actor."""

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False,
      std_min: float = 0.1,
      std_max: float = 0.1,
  ) -> types.Policy:
    del std_min, std_max
    normalizer_params, policy_params = params

    def _actor_observation(observations):
      if isinstance(observations, Mapping) and "actor_state" in observations:
        return observations
      state = _raw_state(observations)
      if tcrmdp_networks.algorithm in (TC_RARL, TC_M2TD3):
        zeros = jnp.zeros(state.shape[:-1] + (tcrmdp_networks.dynamics_param_size,))
        state = jnp.concatenate([state, zeros], axis=-1)
      return {"actor_state": state}

    def deterministic_policy(
        observations: types.Observation,
        key: PRNGKey = None,
    ) -> Tuple[types.Action, types.Extra]:
      del key
      action = tcrmdp_networks.agent_policy_network.apply(
          normalizer_params, policy_params, _actor_observation(observations)
      )
      return action, None

    def stochastic_policy(
        observations: types.Observation,
        noise_scales: jnp.ndarray,
        key: PRNGKey,
    ):
      action = tcrmdp_networks.agent_policy_network.apply(
          normalizer_params, policy_params, _actor_observation(observations)
      )
      noise = jax.random.normal(key, shape=action.shape) * noise_scales[..., None]
      return jnp.clip(action + noise, -1.0, 1.0), None

    return deterministic_policy if deterministic else stochastic_policy

  return make_policy


def make_tcrmdp_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    dynamics_param_size: int,
    algorithm: str = TC_M2TD3,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    **unused_kwargs,
) -> TCRMDPNetworks:
  """Builds agent and adversary networks for TC algorithms."""
  del unused_kwargs
  if algorithm not in (VANILLA_TC_M2TD3, RARL, TC_RARL, TC_M2TD3):
    raise ValueError(f"Unsupported TCRMDP algorithm: {algorithm}")

  agent_policy_network = networks.make_deterministic_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key="actor_state",
  )
  agent_q_network = networks.make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
      obs_key="critic_state",
  )
  adversary_policy_network = networks.make_deterministic_policy_network(
      dynamics_param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key="adv_state",
  )
  adversary_q_network = networks.make_q_network(
      observation_size,
      dynamics_param_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
      obs_key="adv_state",
  )
  return TCRMDPNetworks(
      agent_policy_network=agent_policy_network,
      agent_q_network=agent_q_network,
      adversary_policy_network=adversary_policy_network,
      adversary_q_network=adversary_q_network,
      dynamics_param_size=dynamics_param_size,
      algorithm=algorithm,
  )
