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

"""BridgeTD3 training."""

import functools
import os
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
from brax import base, envs
from brax.envs.base import Env, State
from brax.training import gradients, pmap, replay_buffers, types
from brax.training.acme import running_statistics, specs
from brax.training.types import Metrics, Params, Policy, PRNGKey
import flax
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from agents.bridgetd3 import checkpoint
from agents.bridgetd3 import losses as bridgetd3_losses
from agents.bridgetd3 import networks as bridgetd3_networks
from agents.tcrmdp import common as tc_common
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import AdvEvaluator


ReplayBufferState = Any
_PMAP_AXIS_NAME = "i"


class TransitionwithParams(NamedTuple):
  """Transition with sampled dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  adversary_kinetic: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: FrozenDict[str, Any]


@flax.struct.dataclass
class TrainingState:
  """Contains training state for BridgeTD3."""

  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  adversary_optimizer_state: optax.OptState
  adversary_params: Params
  alpha_optimizer_state: optax.OptState
  bridge_log_alpha: jnp.ndarray
  gradient_steps: types.UInt64
  env_steps: types.UInt64
  normalizer_params: running_statistics.RunningStatisticsState
  noise_scales: jnp.ndarray


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _replicate_across_devices(value, local_devices_to_use: int):
  return jax.device_put(
      jax.tree_util.tree_map(
          lambda x: jnp.broadcast_to(
              jnp.asarray(x), (local_devices_to_use,) + jnp.asarray(x).shape
          ),
          value,
      )
  )


def _uint64_mod(step: types.UInt64, divisor: int) -> jax.Array:
  hi_mod = step.hi % divisor
  lo_mod = step.lo % divisor
  word_mod = (2**32) % divisor
  return (hi_mod * word_mod + lo_mod) % divisor


def _init_training_state(
    key: PRNGKey,
    obs_size: Union[int, Dict[str, specs.Array]],
    local_devices_to_use: int,
    bridgetd3_network: bridgetd3_networks.BridgeTd3Networks,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    adversary_optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    num_envs: int,
    init_log_alpha: float,
    std_max: float = 0.4,
    std_min: float = 0.05,
) -> TrainingState:
  key_policy, key_q, key_adv, key_noise = jax.random.split(key, 4)

  policy_params = bridgetd3_network.policy_network.init(key_policy)
  q_params = bridgetd3_network.q_network.init(key_q)
  adversary_params = bridgetd3_network.adversary_network.init(key_adv)
  bridge_log_alpha = jnp.asarray(init_log_alpha, dtype=jnp.float32)

  normalizer_params = running_statistics.init_state(obs_size)
  training_state = TrainingState(
      policy_optimizer_state=policy_optimizer.init(policy_params),
      policy_params=policy_params,
      q_optimizer_state=q_optimizer.init(q_params),
      q_params=q_params,
      target_q_params=q_params,
      adversary_optimizer_state=adversary_optimizer.init(adversary_params),
      adversary_params=adversary_params,
      alpha_optimizer_state=alpha_optimizer.init(bridge_log_alpha),
      bridge_log_alpha=bridge_log_alpha,
      gradient_steps=types.UInt64(hi=0, lo=0),
      env_steps=types.UInt64(hi=0, lo=0),
      normalizer_params=normalizer_params,
      noise_scales=jax.random.uniform(
          key_noise,
          (num_envs // local_devices_to_use // jax.process_count(),),
          minval=std_min,
          maxval=std_max,
      ),
  )
  return _replicate_across_devices(training_state, local_devices_to_use)


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 1024,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    network_factory: types.NetworkFactory[
        bridgetd3_networks.BridgeTd3Networks
    ] = bridgetd3_networks.make_bridgetd3_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    eval_randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    dr_train_ratio: float = 1.0,
    std_max: float = 0.4,
    std_min: float = 0.05,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_frequency: int = 2,
    dr_augmented_critic: bool = True,
    bridge_alpha: float = 1.0,
    bridge_auto_alpha: bool = True,
    bridge_target_kinetic_coef: float = 2.5,
    bridge_init_log_alpha: Optional[float] = None,
    bridge_alpha_lr: Optional[float] = None,
    adversary_learning_rate: Optional[float] = None,
    use_wandb: bool = False,
    use_tc: bool = False,
    radius: float = 0.001,
):
  """Trains BridgeTD3."""
  if not dr_augmented_critic:
    raise ValueError(
        "BridgeTD3 requires dr_augmented_critic=true for Q(s, a, omega)."
    )
  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  logging.info(
      "local_device_count: %s; total_device_count: %s",
      local_devices_to_use,
      device_count,
  )

  if min_replay_size >= num_timesteps:
    raise ValueError(
        "No training will happen because min_replay_size >= num_timesteps"
    )
  if policy_frequency < 1:
    raise ValueError("policy_frequency must be >= 1")

  if max_replay_size is None:
    max_replay_size = num_timesteps

  env_steps_per_actor_step = action_repeat * num_envs
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )

  assert num_envs % device_count == 0
  import copy

  env = copy.deepcopy(environment)
  rng = jax.random.PRNGKey(seed)
  rng, global_key = jax.random.split(rng)

  obs_shape = env.observation_size
  action_size = env.action_size
  if randomization_fn is None:
    raise ValueError("BridgeTD3 requires randomization=true.")
  if not hasattr(env, "dr_range"):
    raise ValueError("BridgeTD3 requires an environment with dr_range.")
  dr_range_low, dr_range_high = env.dr_range
  dr_mid = (dr_range_low + dr_range_high) / 2.0
  dr_scale = (dr_range_high - dr_range_low) / 2.0
  training_dr_range = (
      dr_mid - dr_train_ratio * dr_scale,
      dr_mid + dr_train_ratio * dr_scale,
  )
  dr_range_low, dr_range_high = training_dr_range
  training_randomization_fn = functools.partial(
      randomization_fn,
      dr_range=training_dr_range,
  )
  env = wrap_for_adv_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=training_randomization_fn,
      param_size=len(dr_range_low),
      dr_range_low=dr_range_low,
      dr_range_high=dr_range_high,
  )

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  bridgetd3_network = network_factory(
      observation_size=obs_shape,
      action_size=action_size,
      param_size=len(dr_range_low),
      preprocess_observations_fn=normalize_fn,
  )
  make_policy = bridgetd3_networks.make_inference_fn(bridgetd3_network)

  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)
  adversary_optimizer = optax.adam(
      learning_rate=learning_rate
      if adversary_learning_rate is None
      else adversary_learning_rate
  )
  alpha_optimizer = optax.adam(
      learning_rate=(
          learning_rate * 0.1
          if bridge_alpha_lr is None
          else bridge_alpha_lr
      )
  )
  if bridge_init_log_alpha is None:
    bridge_init_log_alpha = float(jnp.log(jnp.maximum(bridge_alpha, 1e-8)))
  target_kinetic = bridge_target_kinetic_coef * len(dr_range_low)

  dummy_obs = (
      {key: jnp.zeros(obs_shape[key]) for key in obs_shape}
      if isinstance(obs_shape, dict)
      else jnp.zeros((obs_shape,))
  )
  dummy_action = jnp.zeros((action_size,))
  dummy_params = jnp.zeros((len(dr_range_low),))
  dummy_transition = TransitionwithParams(
      observation=dummy_obs,
      dynamics_params=dummy_params,
      adversary_kinetic=0.0,
      action=dummy_action,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      extras={"state_extras": {"truncation": 0.0}, "policy_extras": {}},
  )
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  critic_loss, actor_loss, adversary_loss, alpha_loss = bridgetd3_losses.make_losses(
      bridgetd3_network=bridgetd3_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      dr_low=dr_range_low,
      dr_high=dr_range_high,
      target_kinetic=target_kinetic,
      use_tc=use_tc,
      radius=radius,
  )
  critic_update = gradients.gradient_update_fn(
      critic_loss, q_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(
      actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )
  adversary_update = gradients.gradient_update_fn(
      adversary_loss,
      adversary_optimizer,
      has_aux=True,
      pmap_axis_name=_PMAP_AXIS_NAME,
  )
  alpha_update = gradients.gradient_update_fn(
      alpha_loss,
      alpha_optimizer,
      pmap_axis_name=_PMAP_AXIS_NAME,
  )

  def sample_adversary_params(
      normalizer_params,
      adversary_params,
      observations,
      key,
      initial_params=None,
      tc_radius=None,
      deterministic: bool = False,
      return_trajectory: bool = False,
  ):
    return bridgetd3_network.adversary_network.apply(
        normalizer_params,
        adversary_params,
        observations,
        key,
        dr_range_low,
        dr_range_high,
        initial_params=initial_params,
        tc_radius=tc_radius,
        deterministic=deterministic,
        return_trajectory=return_trajectory,
    )

  def current_env_params(env_state: envs.State, key: PRNGKey) -> jax.Array:
    reset_params = jax.random.uniform(
        key,
        shape=env_state.info["dr_params"].shape,
        minval=dr_range_low,
        maxval=dr_range_high,
    )
    done = env_state.done[..., None]
    return env_state.info["dr_params"] * (1 - done) + reset_params * done

  def plot_adversary_flow_trajectories(
      training_state: TrainingState,
      env_state: envs.State,
      current_step: int,
      key: PRNGKey,
      num_trajectories: int = 50,
  ) -> list[str]:
    unpmapped_training_state = _unpmap(training_state)
    unpmapped_env_state = _unpmap(env_state)
    obs = unpmapped_env_state.obs
    sample_count = min(
        num_trajectories,
        jax.tree_util.tree_leaves(obs)[0].shape[0],
    )
    obs = jax.tree_util.tree_map(lambda x: x[:sample_count], obs)
    initial_params = None
    if use_tc:
      initial_params = current_env_params(unpmapped_env_state, key)[:sample_count]
    (
        _params,
        _kinetic,
        _latent,
        trajectory_times,
        trajectory_params,
    ) = sample_adversary_params(
        unpmapped_training_state.normalizer_params,
        unpmapped_training_state.adversary_params,
        obs,
        key,
        initial_params=initial_params,
        tc_radius=radius if use_tc else None,
        deterministic=False,
        return_trajectory=True,
    )
    trajectory_times = np.asarray(jax.device_get(trajectory_times))
    trajectory_params = np.asarray(jax.device_get(trajectory_params))
    num_dims = trajectory_params.shape[-1]
    nfe = max(int(trajectory_times.shape[0] - 1), 0)
    plot_dir = os.path.join(os.getcwd(), "bridgetd3_eval_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_paths = []
    for dim in range(num_dims):
      fig, ax = plt.subplots(figsize=(10, 4))
      for idx in range(sample_count):
        ax.plot(
            trajectory_times,
            trajectory_params[idx, :, dim],
            alpha=0.65,
            linewidth=1.5,
        )
        ax.scatter(
            trajectory_times[0],
            trajectory_params[idx, 0, dim],
            color="tab:green",
            s=18,
        )
        ax.scatter(
            trajectory_times[-1],
            trajectory_params[idx, -1, dim],
            color="tab:red",
            s=18,
        )
      ax.set_ylabel(f"dim {dim}")
      ax.set_xlabel("time")
      ax.grid(alpha=0.25)
      ax.set_title(
          "BridgeTD3 adversary flow trajectory "
          f"dim {dim} at step {current_step} (NFE={nfe})"
      )
      fig.tight_layout()
      plot_path = os.path.join(
          plot_dir,
          f"adversary_flow_trajectory_dim_{dim}_step_{current_step}.png",
      )
      fig.savefig(plot_path, dpi=180, bbox_inches="tight")
      plot_paths.append(plot_path)
      if use_wandb:
        import wandb

        wandb.log(
            {
                f"eval/adversary_flow_trajectory_dim_{dim}": wandb.Image(fig),
            },
            step=current_step,
        )
      plt.close(fig)
    return plot_paths

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey],
      transitions: TransitionwithParams,
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry
    key, key_critic, key_actor, key_adv, key_noise = jax.random.split(key, 5)
    noise = jax.random.normal(key_noise, shape=transitions.action.shape) * policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    (
        critic_loss_value,
        (current_q, next_v),
    ), q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.policy_params,
        training_state.adversary_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        transitions,
        noise,
        key_critic,
        optimizer_state=training_state.q_optimizer_state,
    )

    def polyak_update(target_params, params):
      return jax.tree_util.tree_map(
          lambda x, y: x * (1 - tau) + y * tau,
          target_params,
          params,
      )

    new_gradient_steps = training_state.gradient_steps + 1
    should_update_actor = _uint64_mod(new_gradient_steps, policy_frequency) == 0
    new_target_q_params = polyak_update(training_state.target_q_params, q_params)
    current_bridge_alpha = jax.lax.cond(
        bridge_auto_alpha,
        lambda _: jnp.exp(training_state.bridge_log_alpha),
        lambda _: jnp.asarray(bridge_alpha, dtype=jnp.float32),
        operand=None,
    )

    def update_actor_and_adversary(_):
      (
          adversary_loss_value,
          adversary_aux,
      ), adversary_params, adversary_optimizer_state = adversary_update(
          training_state.adversary_params,
          training_state.normalizer_params,
          training_state.policy_params,
          q_params,
          current_bridge_alpha,
          transitions,
          key_adv,
          optimizer_state=training_state.adversary_optimizer_state,
      )
      if bridge_auto_alpha:
        alpha_loss_value, bridge_log_alpha, alpha_optimizer_state = alpha_update(
            training_state.bridge_log_alpha,
            adversary_aux["adversary_kinetic_mean"],
            optimizer_state=training_state.alpha_optimizer_state,
        )
      else:
        alpha_loss_value = jnp.zeros_like(adversary_loss_value)
        bridge_log_alpha = training_state.bridge_log_alpha
        alpha_optimizer_state = training_state.alpha_optimizer_state
      actor_loss_value, policy_params, policy_optimizer_state = actor_update(
          training_state.policy_params,
          training_state.normalizer_params,
          q_params,
          adversary_params,
          transitions,
          key_actor,
          optimizer_state=training_state.policy_optimizer_state,
      )
      return (
          actor_loss_value,
          adversary_loss_value,
          alpha_loss_value,
          adversary_aux,
          policy_params,
          policy_optimizer_state,
          adversary_params,
          adversary_optimizer_state,
          bridge_log_alpha,
          alpha_optimizer_state,
      )

    def skip_actor_and_adversary(_):
      zero = jnp.zeros_like(critic_loss_value)
      adversary_aux = {
          "adversary_kinetic_mean": zero,
          "adversary_q_mean": zero,
          "adversary_params_mean": zero,
          "adversary_params_std": zero,
      }
      return (
          zero,
          zero,
          zero,
          adversary_aux,
          training_state.policy_params,
          training_state.policy_optimizer_state,
          training_state.adversary_params,
          training_state.adversary_optimizer_state,
          training_state.bridge_log_alpha,
          training_state.alpha_optimizer_state,
      )

    (
        actor_loss_value,
        adversary_loss_value,
        alpha_loss_value,
        adversary_aux,
        policy_params,
        policy_optimizer_state,
        adversary_params,
        adversary_optimizer_state,
        bridge_log_alpha,
        alpha_optimizer_state,
    ) = jax.lax.cond(
        should_update_actor,
        update_actor_and_adversary,
        skip_actor_and_adversary,
        operand=None,
    )

    metrics = {
        "critic_loss": critic_loss_value,
        "actor_loss": actor_loss_value,
        "actor_updated": should_update_actor.astype(jnp.float32),
        "adversary_loss": adversary_loss_value,
        "alpha_loss": alpha_loss_value,
        "bridge_alpha": jax.lax.cond(
            bridge_auto_alpha,
            lambda _: jnp.exp(bridge_log_alpha),
            lambda _: jnp.asarray(bridge_alpha, dtype=jnp.float32),
            operand=None,
        ),
        "target_kinetic": jnp.asarray(target_kinetic, dtype=jnp.float32),
        "current_q_min": current_q.min(),
        "current_q_max": current_q.max(),
        "current_q_mean": current_q.mean(),
        "next_v_min": next_v.min(),
        "next_v_max": next_v.max(),
        "next_v_mean": next_v.mean(),
        **adversary_aux,
    }

    new_training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        adversary_optimizer_state=adversary_optimizer_state,
        adversary_params=adversary_params,
        alpha_optimizer_state=alpha_optimizer_state,
        bridge_log_alpha=bridge_log_alpha,
        gradient_steps=new_gradient_steps,
        env_steps=training_state.env_steps,
        normalizer_params=training_state.normalizer_params,
        noise_scales=training_state.noise_scales,
    )
    return (new_training_state, key), metrics

  def adv_step(
      env: Env,
      env_state: State,
      normalizer_params,
      policy_params,
      adversary_params,
      noise_scales: jnp.ndarray,
      key: PRNGKey,
      extra_fields: Sequence[str] = (),
      fixed_dynamics_params: Optional[jnp.ndarray] = None,
  ):
    action_key, adversary_key = jax.random.split(key)
    policy = make_policy((normalizer_params, policy_params))
    actions, policy_extras = policy(env_state.obs, noise_scales, action_key)
    current_params = current_env_params(env_state, adversary_key)
    if fixed_dynamics_params is None:
      candidate_params, adversary_kinetic, _ = sample_adversary_params(
          normalizer_params,
          adversary_params,
          env_state.obs,
          adversary_key,
          initial_params=current_params if use_tc else None,
          tc_radius=radius if use_tc else None,
          deterministic=False,
      )
    else:
      candidate_params = fixed_dynamics_params
      adversary_kinetic = jnp.zeros(env_state.reward.shape, dtype=env_state.reward.dtype)

    if use_tc:
      dynamics_params = tc_common.clip_params_to_radius(
          current_params,
          candidate_params,
          dr_range_low,
          dr_range_high,
          radius,
      )
    else:
      dynamics_params = candidate_params

    nstate = env.step(env_state, actions, dynamics_params)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, TransitionwithParams(
        observation=env_state.obs,
        dynamics_params=dynamics_params,
        adversary_kinetic=adversary_kinetic,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      adversary_params: Params,
      noise_scales: jnp.ndarray,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
      fixed_dynamics_params: Optional[jnp.ndarray] = None,
  ):
    noise_key, key = jax.random.split(key)
    env_state, transitions = adv_step(
        env,
        env_state,
        normalizer_params,
        policy_params,
        adversary_params,
        noise_scales,
        key,
        extra_fields=("truncation",),
        fixed_dynamics_params=fixed_dynamics_params,
    )
    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
    noise_scales = (
        (1 - env_state.done) * noise_scales
        + env_state.done
        * jax.random.uniform(
            noise_key,
            shape=noise_scales.shape,
            minval=std_min,
            maxval=std_max,
        )
    )
    simul_info = {
        "simul/reward_mean": transitions.reward.mean(),
        "simul/reward_std": transitions.reward.std(),
        "simul/reward_max": transitions.reward.max(),
        "simul/reward_min": transitions.reward.min(),
        "simul/dynamics_params_mean": transitions.dynamics_params.mean(),
        "simul/dynamics_params_std": transitions.dynamics_params.std(),
        "simul/adversary_kinetic_mean": transitions.adversary_kinetic.mean(),
    }
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, noise_scales, env_state, buffer_state, simul_info

  def training_step(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    experience_key, training_key = jax.random.split(key)
    (
        normalizer_params,
        noise_scales,
        env_state,
        buffer_state,
        simul_info,
    ) = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.adversary_params,
        training_state.noise_scales,
        env_state,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        noise_scales=noise_scales,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
    )

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )
    metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
    metrics.update(simul_info)
    return training_state, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    def f(carry, params):
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      (
          new_normalizer_params,
          new_noise_scales,
          env_state,
          buffer_state,
          _,
      ) = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          training_state.adversary_params,
          training_state.noise_scales,
          env_state,
          buffer_state,
          key,
          fixed_dynamics_params=params,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          noise_scales=new_noise_scales,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), ()

    param_key, key = jax.random.split(key)
    local_envs = num_envs // jax.process_count() // local_devices_to_use
    dynamics_params = jax.random.uniform(
        param_key,
        shape=(num_prefill_actor_steps, local_envs, len(dr_range_low)),
        minval=dr_range_low,
        maxval=dr_range_high,
    )
    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        dynamics_params,
        length=num_prefill_actor_steps,
    )[0]

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
  )

  def training_epoch(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    def f(carry, unused_t):
      del unused_t
      ts, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, metrics = training_step(ts, es, bs, k)
      return (ts, es, bs, new_key), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, env_state, buffer_state, metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  training_walltime = 0.0

  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    nonlocal training_walltime
    t = time.time()
    training_state, env_state, buffer_state, metrics = training_epoch(
        training_state, env_state, buffer_state, key
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        env_steps_per_actor_step * num_training_steps_per_epoch
    ) / epoch_training_time
    metrics = {
        "training/sps": sps,
        "training/walltime": training_walltime,
        **{f"training/{name}": value for name, value in metrics.items()},
    }
    return training_state, env_state, buffer_state, metrics

  local_key = jax.random.fold_in(rng, process_id)
  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )
  env_state = jax.pmap(env.reset)(env_keys)
  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
  )

  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_shape,
      local_devices_to_use=local_devices_to_use,
      bridgetd3_network=bridgetd3_network,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      adversary_optimizer=adversary_optimizer,
      alpha_optimizer=alpha_optimizer,
      num_envs=num_envs,
      init_log_alpha=bridge_init_log_alpha,
      std_max=std_max,
      std_min=std_min,
  )
  del global_key

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    training_state = training_state.replace(
        normalizer_params=_replicate_across_devices(
            params[0], local_devices_to_use
        ),
        policy_params=_replicate_across_devices(params[1], local_devices_to_use),
        noise_scales=_replicate_across_devices(params[2], local_devices_to_use),
    )

  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use)
  )

  eval_env = copy.deepcopy(environment)
  evaluation_randomization_fn = eval_randomization_fn or randomization_fn
  if evaluation_randomization_fn is not None:
    eval_dr_low, eval_dr_high = environment.dr_range
    eval_env = wrap_for_adv_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=functools.partial(
            evaluation_randomization_fn,
            dr_range=environment.dr_range,
        ),
        param_size=len(eval_dr_low),
        dr_range_low=eval_dr_low,
        dr_range_high=eval_dr_high,
    )
    evaluator = AdvEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        dr_range_low=eval_dr_low,
        dr_range_high=eval_dr_high,
    )
  else:
    raise ValueError("BridgeTD3 expects evaluation randomization to be available.")

  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap((training_state.normalizer_params, training_state.policy_params)),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  t = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_keys
  )
  replay_size = (
      jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
  )
  logging.info("replay size after prefill %s", replay_size)
  assert replay_size >= min_replay_size
  training_walltime = time.time() - t

  current_step = 0
  metrics = {}
  for _ in range(num_evals_after_init):
    logging.info("step %s", current_step)
    epoch_key, plot_key, local_key = jax.random.split(local_key, 3)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    training_state, env_state, buffer_state, training_metrics = (
        training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
    )
    current_step = int(_unpmap(training_state.env_steps))

    if process_id == 0:
      if checkpoint_logdir:
        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.policy_params,
                training_state.noise_scales,
            )
        )
        ckpt_config = checkpoint.network_config(
            observation_size=obs_shape,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)

      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)
          ),
          training_metrics,
      )
      plot_paths = plot_adversary_flow_trajectories(
          training_state,
          env_state,
          current_step,
          plot_key,
      )
      if plot_paths:
        metrics["eval/adversary_flow_plot_paths"] = ",".join(plot_paths)
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if total_steps < num_timesteps:
    raise AssertionError(
      f"Total steps {total_steps} is less than `num_timesteps`= {num_timesteps}."
    )

  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params)
  )
  pmap.assert_is_replicated(training_state)
  logging.info("total steps: %s", total_steps)
  pmap.synchronize_hosts()
  return make_policy, params, metrics
