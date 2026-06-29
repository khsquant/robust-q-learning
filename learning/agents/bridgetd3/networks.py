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

"""BridgeTD3 networks."""

from typing import Mapping, Optional, Sequence, Tuple

import flax
from flax import linen
import jax
import jax.numpy as jnp

from brax.training import types
from brax.training.types import PRNGKey
from module import networks


@flax.struct.dataclass
class BridgeTd3Networks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  adversary_network: networks.FeedForwardNetwork


def _obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]

def _preprocess_obs(
    obs,
    processor_params,
    preprocess_observations_fn,
    obs_key: str,
):
  if isinstance(obs, Mapping):
    obs = obs[obs_key]
    processor_params = networks.normalizer_select(processor_params, obs_key)
  return preprocess_observations_fn(obs, processor_params)


class _VelocityField(linen.Module):
  param_size: int
  hidden_layer_sizes: Sequence[int]
  activation: networks.ActivationFn = linen.relu
  layer_norm: bool = False

  @linen.compact
  def __call__(
      self,
      obs: jnp.ndarray,
      latent: jnp.ndarray,
      time: jnp.ndarray,
  ) -> jnp.ndarray:
    x = jnp.concatenate([obs, latent, time], axis=-1)
    return networks.MLP(
        layer_sizes=list(self.hidden_layer_sizes) + [self.param_size],
        activation=self.activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        layer_norm=self.layer_norm,
    )(x)


class _BridgeFlowAdversary(linen.Module):
  param_size: int
  hidden_layer_sizes: Sequence[int]
  num_flow_steps: int = 4
  activation: networks.ActivationFn = linen.relu
  layer_norm: bool = False
  reference_scale: float = 0.2

  def setup(self):
    self.velocity_field = _VelocityField(
        param_size=self.param_size,
        hidden_layer_sizes=self.hidden_layer_sizes,
        activation=self.activation,
        layer_norm=self.layer_norm,
    )

  def _project_to_bounds(
      self,
      latent: jnp.ndarray,
      dr_low: jnp.ndarray,
      dr_high: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    bounded = jnp.tanh(latent)
    params = dr_low + 0.5 * (bounded + 1.0) * (dr_high - dr_low)
    return params, bounded

  def _params_to_latent(
      self,
      params: jnp.ndarray,
      dr_low: jnp.ndarray,
      dr_high: jnp.ndarray,
  ) -> jnp.ndarray:
    denom = jnp.maximum(dr_high - dr_low, 1e-6)
    bounded = 2.0 * (params - dr_low) / denom - 1.0
    bounded = jnp.clip(bounded, -0.999999, 0.999999)
    return jnp.arctanh(bounded)

  def _clip_local_params(
      self,
      params: jnp.ndarray,
      initial_params: jnp.ndarray,
      dr_low: jnp.ndarray,
      dr_high: jnp.ndarray,
      tc_radius: float,
  ) -> jnp.ndarray:
    local_low = jnp.maximum(dr_low, initial_params - tc_radius)
    local_high = jnp.minimum(dr_high, initial_params + tc_radius)
    return jnp.clip(params, local_low, local_high)

  def _flow_step(
      self,
      obs: jnp.ndarray,
      latent: jnp.ndarray,
      time_start: jnp.ndarray,
      dt: float,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    velocity_start = self.velocity_field(obs, latent, time_start)
    midpoint = latent + 0.5 * dt * velocity_start
    velocity_mid = self.velocity_field(obs, midpoint, time_start + 0.5 * dt)
    next_latent = latent + dt * velocity_mid
    step_energy = 0.5 * jnp.sum(jnp.square(velocity_mid), axis=-1) * dt
    return next_latent, step_energy, midpoint

  @linen.compact
  def __call__(
      self,
      obs: jnp.ndarray,
      rng: PRNGKey,
      dr_low: jnp.ndarray,
      dr_high: jnp.ndarray,
      initial_params: Optional[jnp.ndarray] = None,
      tc_radius: Optional[float] = None,
      deterministic: bool = False,
      return_trajectory: bool = False,
  ):
    batch_size = obs.shape[0]
    num_steps = max(int(self.num_flow_steps), 1)
    dt = 1.0 / num_steps
    dtype = obs.dtype

    if initial_params is not None:
      initial_params = jnp.clip(initial_params, dr_low, dr_high)
      latent = self._params_to_latent(initial_params, dr_low, dr_high)
    elif deterministic:
      latent = jnp.zeros((batch_size, self.param_size), dtype=dtype)
    else:
      latent = jax.random.normal(
          rng,
          shape=(batch_size, self.param_size),
          dtype=dtype,
      ) * self.reference_scale

    time = jnp.zeros((batch_size, 1), dtype=dtype)
    total_kinetic = jnp.zeros((batch_size,), dtype=dtype)
    if return_trajectory:
      initial_params_for_plot, _ = self._project_to_bounds(latent, dr_low, dr_high)
      if initial_params is not None and tc_radius is not None:
        initial_params_for_plot = self._clip_local_params(
            initial_params_for_plot,
            initial_params,
            dr_low,
            dr_high,
            tc_radius,
        )
      trajectory_times = [jnp.asarray(0.0, dtype=dtype)]
      trajectory_params = [jnp.clip(initial_params_for_plot, dr_low, dr_high)]
    for _ in range(num_steps):
      latent, step_energy, midpoint = self._flow_step(obs, latent, time, dt)
      if initial_params is not None and tc_radius is not None:
        next_params, _ = self._project_to_bounds(latent, dr_low, dr_high)
        next_params = self._clip_local_params(
            next_params,
            initial_params,
            dr_low,
            dr_high,
            tc_radius,
        )
        latent = self._params_to_latent(next_params, dr_low, dr_high)
      total_kinetic = total_kinetic + step_energy
      if return_trajectory:
        midpoint_params, _ = self._project_to_bounds(midpoint, dr_low, dr_high)
        final_params, _ = self._project_to_bounds(latent, dr_low, dr_high)
        if initial_params is not None and tc_radius is not None:
          midpoint_params = self._clip_local_params(
              midpoint_params,
              initial_params,
              dr_low,
              dr_high,
              tc_radius,
          )
          final_params = self._clip_local_params(
              final_params,
              initial_params,
              dr_low,
              dr_high,
              tc_radius,
          )
        trajectory_times.append(jnp.asarray(time[0, 0] + 0.5 * dt, dtype=dtype))
        trajectory_params.append(jnp.clip(midpoint_params, dr_low, dr_high))
        trajectory_times.append(jnp.asarray(time[0, 0] + dt, dtype=dtype))
        trajectory_params.append(jnp.clip(final_params, dr_low, dr_high))
      time = time + dt

    params, bounded_latent = self._project_to_bounds(latent, dr_low, dr_high)
    if initial_params is not None and tc_radius is not None:
      params = self._clip_local_params(
          params,
          initial_params,
          dr_low,
          dr_high,
          tc_radius,
      )
    params = jnp.clip(params, dr_low, dr_high)
    if return_trajectory:
      trajectory_times = jnp.stack(trajectory_times, axis=0)
      trajectory_params = jnp.stack(trajectory_params, axis=1)
      return (
          params,
          total_kinetic,
          bounded_latent,
          trajectory_times,
          trajectory_params,
      )
    return params, total_kinetic, bounded_latent


def make_inference_fn(bridgetd3_networks: BridgeTd3Networks):
  """Creates params and inference function for the BridgeTD3 agent."""

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False,
      std_min: float = 0.05,
      std_max: float = 0.8,
  ) -> types.Policy:
    del std_min, std_max

    def deterministic_policy(
        observations: types.Observation,
        key: PRNGKey = None,
    ) -> Tuple[types.Action, types.Extra]:
      del key
      return bridgetd3_networks.policy_network.apply(*params, observations), None

    def stochastic_policy(
        observations: types.Observation,
        noise_scales: jnp.ndarray,
        key: PRNGKey,
    ):
      action = bridgetd3_networks.policy_network.apply(*params, observations)
      noise = jax.random.normal(key, shape=action.shape) * noise_scales[..., None]
      return jnp.clip(action + noise, -1.0, 1.0), None

    return deterministic_policy if deterministic else stochastic_policy

  return make_policy


def make_bridgetd3_networks(
    observation_size: int,
    action_size: int,
    param_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    adversary_hidden_layer_sizes: Optional[Sequence[int]] = None,
    adversary_num_flow_steps: int = 4,
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    adversary_layer_norm: bool = False,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    adversary_obs_key: str = "state",
) -> BridgeTd3Networks:
  """Builds TD3 policy/Q networks plus a FLAC-style dynamics adversary."""
  if adversary_hidden_layer_sizes is None:
    adversary_hidden_layer_sizes = hidden_layer_sizes

  policy_network = networks.make_deterministic_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key=policy_obs_key,
  )
  q_network = networks.make_augmented_q_network(
      observation_size,
      action_size,
      param_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
      obs_key=value_obs_key,
  )

  adversary_module = _BridgeFlowAdversary(
      param_size=param_size,
      hidden_layer_sizes=tuple(adversary_hidden_layer_sizes),
      num_flow_steps=adversary_num_flow_steps,
      activation=activation,
      layer_norm=adversary_layer_norm,
  )

  def adversary_apply(
      processor_params,
      adversary_params,
      obs,
      rng: PRNGKey,
      dr_low: jnp.ndarray,
      dr_high: jnp.ndarray,
      initial_params: Optional[jnp.ndarray] = None,
      tc_radius: Optional[float] = None,
      deterministic: bool = False,
      return_trajectory: bool = False,
  ):
    obs = _preprocess_obs(
        obs,
        processor_params,
        preprocess_observations_fn,
        adversary_obs_key,
    )
    return adversary_module.apply(
        adversary_params,
        obs,
        rng,
        dr_low,
        dr_high,
        initial_params=initial_params,
        tc_radius=tc_radius,
        deterministic=deterministic,
        return_trajectory=return_trajectory,
    )

  obs_size = _obs_state_size(observation_size, adversary_obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  dummy_low = jnp.zeros((param_size,))
  dummy_high = jnp.ones((param_size,))
  dummy_rng = jax.random.PRNGKey(0)
  adversary_network = networks.FeedForwardNetwork(
      init=lambda key: adversary_module.init(
          key,
          dummy_obs,
          dummy_rng,
          dummy_low,
          dummy_high,
          deterministic=False,
      ),
      apply=adversary_apply,
  )

  return BridgeTd3Networks(
      policy_network=policy_network,
      q_network=q_network,
      adversary_network=adversary_network,
  )
