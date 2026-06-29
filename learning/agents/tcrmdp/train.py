# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Training for TCRMDP-style TD3 algorithms."""

import copy
import functools
import time
import struct
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

from absl import logging
from brax import base
from brax import envs
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Metrics, Params, PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax

from agents.tcrmdp import common
from agents.tcrmdp import losses as tcrmdp_losses
from agents.tcrmdp import networks as tcrmdp_networks
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import AdvEvaluator, Evaluator


ReplayBufferState = Any
_PMAP_AXIS_NAME = "i"


class TCTransition(NamedTuple):
  raw_observation: types.Observation
  actor_observation: jax.Array
  critic_observation: jax.Array
  adv_observation: jax.Array
  action: jax.Array
  adv_action: jax.Array
  reward: jax.Array
  adv_reward: jax.Array
  discount: jax.Array
  raw_next_observation: types.Observation
  next_actor_observation: jax.Array
  next_critic_observation: jax.Array
  next_adv_observation: jax.Array
  dynamics_params: jax.Array
  next_dynamics_params: jax.Array
  extras: dict[str, Any]


@flax.struct.dataclass
class TrainingState:
  agent_policy_optimizer_state: optax.OptState
  agent_policy_params: Params
  target_agent_policy_params: Params
  agent_q_optimizer_state: optax.OptState
  agent_q_params: Params
  target_agent_q_params: Params
  adversary_policy_optimizer_state: optax.OptState
  adversary_policy_params: Params
  target_adversary_policy_params: Params
  adversary_q_optimizer_state: optax.OptState
  adversary_q_params: Params
  target_adversary_q_params: Params
  gradient_steps: types.UInt64
  env_steps: types.UInt64
  normalizer_params: running_statistics.RunningStatisticsState
  noise_scales: jax.Array


def _unpmap(value):
  return jax.tree_util.tree_map(lambda x: x[0], value)


def _uint64_mod(step: types.UInt64, divisor: int) -> jax.Array:
  hi_mod = step.hi % divisor
  lo_mod = step.lo % divisor
  word_mod = (2**32) % divisor
  return (hi_mod * word_mod + lo_mod) % divisor


def _zeros_from_spec(spec):
  if isinstance(spec, dict):
    return {k: _zeros_from_spec(v) for k, v in spec.items()}
  return jnp.zeros(spec.shape, dtype=jnp.float32)


def _spec_from_pmap_obs(obs):
  return jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), obs
  )


def _base_obs_size(obs_spec) -> int:
  if isinstance(obs_spec, dict):
    return obs_spec["state"].shape[-1]
  return obs_spec.shape[-1]


def _init_training_state(
    key: PRNGKey,
    obs_size: dict[str, specs.Array],
    local_devices_to_use: int,
    tcrmdp_network: tcrmdp_networks.TCRMDPNetworks,
    agent_policy_optimizer: optax.GradientTransformation,
    agent_q_optimizer: optax.GradientTransformation,
    adversary_policy_optimizer: optax.GradientTransformation,
    adversary_q_optimizer: optax.GradientTransformation,
    num_envs: int,
    std_max: float,
    std_min: float,
) -> TrainingState:
  key_agent_policy, key_agent_q, key_adv_policy, key_adv_q, key_noise = (
      jax.random.split(key, 5)
  )
  agent_policy_params = tcrmdp_network.agent_policy_network.init(key_agent_policy)
  agent_q_params = tcrmdp_network.agent_q_network.init(key_agent_q)
  adversary_policy_params = tcrmdp_network.adversary_policy_network.init(
      key_adv_policy
  )
  adversary_q_params = tcrmdp_network.adversary_q_network.init(key_adv_q)

  training_state = TrainingState(
      agent_policy_optimizer_state=agent_policy_optimizer.init(
          agent_policy_params
      ),
      agent_policy_params=agent_policy_params,
      target_agent_policy_params=agent_policy_params,
      agent_q_optimizer_state=agent_q_optimizer.init(agent_q_params),
      agent_q_params=agent_q_params,
      target_agent_q_params=agent_q_params,
      adversary_policy_optimizer_state=adversary_policy_optimizer.init(
          adversary_policy_params
      ),
      adversary_policy_params=adversary_policy_params,
      target_adversary_policy_params=adversary_policy_params,
      adversary_q_optimizer_state=adversary_q_optimizer.init(adversary_q_params),
      adversary_q_params=adversary_q_params,
      target_adversary_q_params=adversary_q_params,
      gradient_steps=types.UInt64(hi=0, lo=0),
      env_steps=types.UInt64(hi=0, lo=0),
      normalizer_params=running_statistics.init_state(obs_size),
      noise_scales=jax.random.uniform(
          key_noise,
          (num_envs // local_devices_to_use // jax.process_count(),),
          minval=std_min,
          maxval=std_max,
      ),
  )
  def replicate(x):
    x = jnp.asarray(x)
    return jax.device_put(jnp.broadcast_to(x, (local_devices_to_use,) + x.shape))

  return jax.tree_util.tree_map(replicate, training_state)


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 1024,
    learning_rate: float = 3e-4,
    discounting: float = 0.99,
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
        tcrmdp_networks.TCRMDPNetworks
    ] = tcrmdp_networks.make_tcrmdp_networks,
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
    std_max: float = 0.1,
    std_min: float = 0.1,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_frequency: int = 2,
    radius: float = 0.001,
    rarl_range_scale: float = 0.5,
    omniscient_adversary: bool = True,
    asymmetric_critic: bool = True,
    dr_augmented_critic: bool = False,
    algorithm: str = tcrmdp_networks.TC_M2TD3,
    **unused_kwargs,
):
  """Trains one of the TCRMDP-style algorithms."""
  del checkpoint_logdir, restore_checkpoint_path, unused_kwargs
  if algorithm not in (
      tcrmdp_networks.VANILLA_TC_M2TD3,
      tcrmdp_networks.RARL,
      tcrmdp_networks.TC_RARL,
      tcrmdp_networks.TC_M2TD3,
  ):
    raise ValueError(f"Unsupported TCRMDP algorithm: {algorithm}")
  if randomization_fn is None:
    raise ValueError(f"{algorithm} requires randomization=true.")
  if not hasattr(environment, "dr_range"):
    raise ValueError(f"{algorithm} requires environments with dr_range.")
  if policy_frequency < 1:
    raise ValueError("policy_frequency must be >= 1")
  if min_replay_size >= num_timesteps:
    raise ValueError(
        "No training will happen because min_replay_size >= num_timesteps"
    )
  if max_replay_size is None:
    max_replay_size = num_timesteps

  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  assert num_envs % device_count == 0

  env_steps_per_actor_step = action_repeat * num_envs
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )

  env = copy.deepcopy(environment)
  dr_low, dr_high = env.dr_range
  dr_mid = (dr_low + dr_high) / 2.0
  dr_scale = (dr_high - dr_low) / 2.0
  training_dr_range = (
      dr_mid - dr_train_ratio * dr_scale,
      dr_mid + dr_train_ratio * dr_scale,
  )
  dr_low, dr_high = training_dr_range
  dynamics_param_size = len(dr_low)

  env = wrap_for_adv_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=functools.partial(
          randomization_fn, dr_range=training_dr_range
      ),
      param_size=dynamics_param_size,
      dr_range_low=dr_low,
      dr_range_high=dr_high,
  )

  rng = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(rng)
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )
  env_state = jax.pmap(env.reset)(env_keys)
  raw_obs_spec = _spec_from_pmap_obs(env_state.obs)
  raw_dummy_obs = _zeros_from_spec(raw_obs_spec)
  base_obs_size = _base_obs_size(raw_obs_spec)
  if algorithm == tcrmdp_networks.RARL:
    actor_obs_size = base_obs_size
    critic_obs_size = base_obs_size
    adv_obs_size = base_obs_size + (
        env.action_size if omniscient_adversary else 0
    )
  else:
    actor_obs_size = base_obs_size
    if dr_augmented_critic:
      critic_obs_size = base_obs_size
    else:
      critic_obs_size = (
          base_obs_size + dynamics_param_size
          if asymmetric_critic
          else actor_obs_size
      )
    adv_obs_size = base_obs_size + dynamics_param_size + env.action_size
  tc_obs_size = {
      "actor_state": specs.Array((actor_obs_size,), jnp.dtype("float32")),
      "critic_state": specs.Array((critic_obs_size,), jnp.dtype("float32")),
      "adv_state": specs.Array((adv_obs_size,), jnp.dtype("float32")),
  }
  tc_network_obs_size = {
      "actor_state": actor_obs_size,
      "critic_state": critic_obs_size,
      "adv_state": adv_obs_size,
  }

  normalize_fn = running_statistics.normalize if normalize_observations else (
      lambda x, y: x
  )
  tcrmdp_network = network_factory(
      observation_size=tc_network_obs_size,
      action_size=env.action_size,
      dynamics_param_size=dynamics_param_size,
      algorithm=algorithm,
      preprocess_observations_fn=normalize_fn,
      dr_augmented_critic=dr_augmented_critic,
  )
  make_policy = tcrmdp_networks.make_inference_fn(tcrmdp_network)

  agent_policy_optimizer = optax.adam(learning_rate=learning_rate)
  agent_q_optimizer = optax.adam(learning_rate=learning_rate)
  adversary_policy_optimizer = optax.adam(learning_rate=learning_rate)
  adversary_q_optimizer = optax.adam(learning_rate=learning_rate)

  dummy_action = jnp.zeros((env.action_size,), dtype=jnp.float32)
  dummy_adv_action = jnp.zeros((dynamics_param_size,), dtype=jnp.float32)
  dummy_actor_obs = jnp.zeros((actor_obs_size,), dtype=jnp.float32)
  dummy_critic_obs = jnp.zeros((critic_obs_size,), dtype=jnp.float32)
  dummy_adv_obs = jnp.zeros((adv_obs_size,), dtype=jnp.float32)
  dummy_params = jnp.zeros((dynamics_param_size,), dtype=jnp.float32)
  dummy_transition = TCTransition(
      raw_observation=raw_dummy_obs,
      actor_observation=dummy_actor_obs,
      critic_observation=dummy_critic_obs,
      adv_observation=dummy_adv_obs,
      action=dummy_action,
      adv_action=dummy_adv_action,
      reward=0.0,
      adv_reward=0.0,
      discount=0.0,
      raw_next_observation=raw_dummy_obs,
      next_actor_observation=dummy_actor_obs,
      next_critic_observation=dummy_critic_obs,
      next_adv_observation=dummy_adv_obs,
      dynamics_params=dummy_params,
      next_dynamics_params=dummy_params,
      extras={"state_extras": {"truncation": 0.0}, "policy_extras": {}},
  )
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  (
      agent_critic_loss,
      agent_actor_loss,
      rarl_adversary_critic_loss,
      rarl_adversary_actor_loss,
      m2td3_adversary_actor_loss,
  ) = tcrmdp_losses.make_losses(
      tcrmdp_network=tcrmdp_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      dr_low=dr_low,
      dr_high=dr_high,
      radius=radius,
      omniscient_adversary=omniscient_adversary,
      asymmetric_critic=asymmetric_critic,
      dr_augmented_critic=dr_augmented_critic,
  )
  agent_critic_update = gradients.gradient_update_fn(
      agent_critic_loss,
      agent_q_optimizer,
      has_aux=True,
      pmap_axis_name=_PMAP_AXIS_NAME,
  )
  agent_actor_update = gradients.gradient_update_fn(
      agent_actor_loss, agent_policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )
  adversary_critic_update = gradients.gradient_update_fn(
      rarl_adversary_critic_loss,
      adversary_q_optimizer,
      has_aux=True,
      pmap_axis_name=_PMAP_AXIS_NAME,
  )
  rarl_adversary_actor_update = gradients.gradient_update_fn(
      rarl_adversary_actor_loss,
      adversary_policy_optimizer,
      pmap_axis_name=_PMAP_AXIS_NAME,
  )
  m2td3_adversary_actor_update = gradients.gradient_update_fn(
      m2td3_adversary_actor_loss,
      adversary_policy_optimizer,
      pmap_axis_name=_PMAP_AXIS_NAME,
  )

  def polyak_update(target_params, params):
    return jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, target_params, params
    )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: TCTransition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry
    key, key_agent_critic, key_agent_actor, key_adv_critic, key_adv_actor, key_noise = (
        jax.random.split(key, 6)
    )
    action_noise = jnp.clip(
        jax.random.normal(key_noise, shape=transitions.action.shape)
        * policy_noise,
        -noise_clip,
        noise_clip,
    )
    (
        agent_critic_loss_value,
        (agent_current_q, agent_next_v),
    ), agent_q_params, agent_q_optimizer_state = agent_critic_update(
        training_state.agent_q_params,
        training_state.target_agent_policy_params,
        training_state.normalizer_params,
        training_state.target_agent_q_params,
        transitions,
        action_noise,
        key_agent_critic,
        optimizer_state=training_state.agent_q_optimizer_state,
    )

    if algorithm in (tcrmdp_networks.RARL, tcrmdp_networks.TC_RARL):
      adv_noise = jnp.clip(
          jax.random.normal(key_noise, shape=transitions.adv_action.shape)
          * policy_noise,
          -noise_clip,
          noise_clip,
      )
      (
          adversary_critic_loss_value,
          (adversary_current_q, adversary_next_v),
      ), adversary_q_params, adversary_q_optimizer_state = adversary_critic_update(
          training_state.adversary_q_params,
          training_state.target_adversary_policy_params,
          training_state.normalizer_params,
          training_state.target_adversary_q_params,
          transitions,
          adv_noise,
          key_adv_critic,
          optimizer_state=training_state.adversary_q_optimizer_state,
      )
    else:
      adversary_critic_loss_value = jnp.zeros_like(agent_critic_loss_value)
      adversary_current_q = jnp.zeros_like(agent_current_q)
      adversary_next_v = jnp.zeros_like(agent_next_v)
      adversary_q_params = training_state.adversary_q_params
      adversary_q_optimizer_state = training_state.adversary_q_optimizer_state

    new_target_agent_q_params = polyak_update(
        training_state.target_agent_q_params, agent_q_params
    )
    if algorithm in (tcrmdp_networks.RARL, tcrmdp_networks.TC_RARL):
      new_target_adversary_q_params = polyak_update(
          training_state.target_adversary_q_params, adversary_q_params
      )
    else:
      new_target_adversary_q_params = training_state.target_adversary_q_params

    def update_actors_and_targets(_):
      agent_actor_loss_value, agent_policy_params, agent_policy_optimizer_state = (
          agent_actor_update(
              training_state.agent_policy_params,
              training_state.normalizer_params,
              agent_q_params,
              transitions,
              key_agent_actor,
              optimizer_state=training_state.agent_policy_optimizer_state,
          )
      )
      if algorithm in (tcrmdp_networks.RARL, tcrmdp_networks.TC_RARL):
        (
            adversary_actor_loss_value,
            adversary_policy_params,
            adversary_policy_optimizer_state,
        ) = rarl_adversary_actor_update(
            training_state.adversary_policy_params,
            training_state.normalizer_params,
            adversary_q_params,
            transitions,
            key_adv_actor,
            optimizer_state=training_state.adversary_policy_optimizer_state,
        )
      else:
        (
            adversary_actor_loss_value,
            adversary_policy_params,
            adversary_policy_optimizer_state,
        ) = m2td3_adversary_actor_update(
            training_state.adversary_policy_params,
            training_state.normalizer_params,
            agent_q_params,
            transitions,
            key_adv_actor,
            optimizer_state=training_state.adversary_policy_optimizer_state,
        )

      return (
          agent_actor_loss_value,
          adversary_actor_loss_value,
          agent_policy_params,
          agent_policy_optimizer_state,
          polyak_update(training_state.target_agent_policy_params, agent_policy_params),
          adversary_policy_params,
          adversary_policy_optimizer_state,
          polyak_update(
              training_state.target_adversary_policy_params,
              adversary_policy_params,
          ),
      )

    def skip_actors_and_targets(_):
      return (
          jnp.zeros_like(agent_critic_loss_value),
          jnp.zeros_like(agent_critic_loss_value),
          training_state.agent_policy_params,
          training_state.agent_policy_optimizer_state,
          training_state.target_agent_policy_params,
          training_state.adversary_policy_params,
          training_state.adversary_policy_optimizer_state,
          training_state.target_adversary_policy_params,
      )

    new_gradient_steps = training_state.gradient_steps + 1
    should_update_actor = _uint64_mod(new_gradient_steps, policy_frequency) == 0
    (
        agent_actor_loss_value,
        adversary_actor_loss_value,
        agent_policy_params,
        agent_policy_optimizer_state,
        target_agent_policy_params,
        adversary_policy_params,
        adversary_policy_optimizer_state,
        target_adversary_policy_params,
    ) = jax.lax.cond(
        should_update_actor,
        update_actors_and_targets,
        skip_actors_and_targets,
        operand=None,
    )

    metrics = {
        "agent_critic_loss": agent_critic_loss_value,
        "agent_actor_loss": agent_actor_loss_value,
        "adversary_critic_loss": adversary_critic_loss_value,
        "adversary_actor_loss": adversary_actor_loss_value,
        "actor_updated": should_update_actor.astype(jnp.float32),
        "agent_current_q_mean": agent_current_q.mean(),
        "agent_next_v_mean": agent_next_v.mean(),
        "adversary_current_q_mean": adversary_current_q.mean(),
        "adversary_next_v_mean": adversary_next_v.mean(),
    }
    new_training_state = training_state.replace(
        agent_policy_optimizer_state=agent_policy_optimizer_state,
        agent_policy_params=agent_policy_params,
        target_agent_policy_params=target_agent_policy_params,
        agent_q_optimizer_state=agent_q_optimizer_state,
        agent_q_params=agent_q_params,
        target_agent_q_params=new_target_agent_q_params,
        adversary_policy_optimizer_state=adversary_policy_optimizer_state,
        adversary_policy_params=adversary_policy_params,
        target_adversary_policy_params=target_adversary_policy_params,
        adversary_q_optimizer_state=adversary_q_optimizer_state,
        adversary_q_params=adversary_q_params,
        target_adversary_q_params=new_target_adversary_q_params,
        gradient_steps=new_gradient_steps,
    )
    return (new_training_state, key), metrics

  def get_experience(
      training_state: TrainingState,
      env_state,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
      random_actions: bool,
  ):
    key, reset_key, action_key, noise_key, adv_key, adv_noise_key = (
        jax.random.split(key, 6)
    )
    reset_params = jax.random.uniform(
        reset_key,
        shape=env_state.info["dr_params"].shape,
        minval=dr_low,
        maxval=dr_high,
    )
    current_params = (
        env_state.info["dr_params"] * (1 - env_state.done[..., None])
        + reset_params * env_state.done[..., None]
    )
    zero_action = jnp.zeros(env_state.reward.shape + (env.action_size,))
    actor_obs, critic_obs, _ = common.build_observations(
        env_state.obs,
        current_params,
        zero_action,
        dr_low,
        dr_high,
        algorithm,
        omniscient_adversary,
        asymmetric_critic,
        dr_augmented_critic,
    )
    if random_actions:
      actions = jax.random.uniform(
          action_key,
          shape=env_state.reward.shape + (env.action_size,),
          minval=-1.0,
          maxval=1.0,
      )
    else:
      actions = tcrmdp_network.agent_policy_network.apply(
          training_state.normalizer_params,
          training_state.agent_policy_params,
          {"actor_state": actor_obs},
      )
      action_noise = (
          jax.random.normal(noise_key, shape=actions.shape)
          * training_state.noise_scales[..., None]
      )
      actions = jnp.clip(actions + action_noise, -1.0, 1.0)

    _, _, adv_obs = common.build_observations(
        env_state.obs,
        current_params,
        actions,
        dr_low,
        dr_high,
        algorithm,
        omniscient_adversary,
        asymmetric_critic,
        dr_augmented_critic,
    )
    if random_actions:
      adv_actions = jax.random.uniform(
          adv_key,
          shape=env_state.reward.shape + (dynamics_param_size,),
          minval=-1.0,
          maxval=1.0,
      )
    else:
      adv_actions = tcrmdp_network.adversary_policy_network.apply(
          training_state.normalizer_params,
          training_state.adversary_policy_params,
          {"adv_state": adv_obs},
      )
      adv_actions = jnp.clip(
          adv_actions
          + jax.random.normal(adv_noise_key, shape=adv_actions.shape)
          * training_state.noise_scales[..., None],
          -1.0,
          1.0,
      )

    if algorithm == tcrmdp_networks.RARL:
      next_params = common.action_to_params(
          adv_actions,
          dr_low,
          dr_high,
          scale=rarl_range_scale,
      )
    else:
      next_params = common.update_params(
          current_params, adv_actions, dr_low, dr_high, radius
      )
    next_env_state = env.step(env_state, actions, next_params)
    next_actor_obs, next_critic_obs, _ = common.build_observations(
        next_env_state.obs,
        next_params,
        zero_action,
        dr_low,
        dr_high,
        algorithm,
        omniscient_adversary,
        asymmetric_critic,
        dr_augmented_critic,
    )
    next_agent_action = tcrmdp_network.agent_policy_network.apply(
        training_state.normalizer_params,
        training_state.agent_policy_params,
        {"actor_state": next_actor_obs},
    )
    _, _, next_adv_obs = common.build_observations(
        next_env_state.obs,
        next_params,
        next_agent_action,
        dr_low,
        dr_high,
        algorithm,
        omniscient_adversary,
        asymmetric_critic,
        dr_augmented_critic,
    )
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        common.observation_dict(actor_obs, critic_obs, adv_obs),
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
    transition = TCTransition(
        raw_observation=env_state.obs,
        actor_observation=actor_obs,
        critic_observation=critic_obs,
        adv_observation=adv_obs,
        action=actions,
        adv_action=adv_actions,
        reward=next_env_state.reward,
        adv_reward=-next_env_state.reward,
        discount=1 - next_env_state.done,
        raw_next_observation=next_env_state.obs,
        next_actor_observation=next_actor_obs,
        next_critic_observation=next_critic_obs,
        next_adv_observation=next_adv_obs,
        dynamics_params=current_params,
        next_dynamics_params=next_params,
        extras={
            "policy_extras": {},
            "state_extras": {"truncation": next_env_state.info["truncation"]},
        },
    )
    noise_scales = (
        (1 - next_env_state.done) * training_state.noise_scales
        + next_env_state.done
        * (
            jax.random.uniform(
                noise_key,
                shape=training_state.noise_scales.shape,
                minval=std_min,
                maxval=std_max,
            )
        )
    )
    buffer_state = replay_buffer.insert(buffer_state, transition)
    simul_info = {
        "simul/reward_mean": transition.reward.mean(),
        "simul/reward_std": transition.reward.std(),
        "simul/params_mean": next_params.mean(),
        "simul/params_std": next_params.std(),
        "simul/adv_action_mean": adv_actions.mean(),
        "simul/adv_action_std": adv_actions.std(),
    }
    return (
        training_state.replace(
            normalizer_params=normalizer_params,
            noise_scales=noise_scales,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        ),
        next_env_state,
        buffer_state,
        simul_info,
    )

  def training_step(
      training_state: TrainingState,
      env_state,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    experience_key, training_key = jax.random.split(key)
    training_state, env_state, buffer_state, simul_info = get_experience(
        training_state, env_state, buffer_state, experience_key, False
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
      env_state,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, key = carry
      key, step_key = jax.random.split(key)
      training_state, env_state, buffer_state, _ = get_experience(
          training_state, env_state, buffer_state, step_key, True
      )
      return (training_state, env_state, buffer_state, key), ()

    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_prefill_actor_steps,
    )[0]

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
  )

  def training_epoch(
      training_state: TrainingState,
      env_state,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    def f(carry, unused_t):
      del unused_t
      ts, es, bs, k = carry
      k, step_key = jax.random.split(k)
      ts, es, bs, metrics = training_step(ts, es, bs, step_key)
      return (ts, es, bs, k), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, env_state, buffer_state, metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    nonlocal training_walltime
    start_time = time.time()
    training_state, env_state, buffer_state, metrics = training_epoch(
        training_state, env_state, buffer_state, key
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
    epoch_training_time = time.time() - start_time
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

  training_state = _init_training_state(
      key=global_key,
      obs_size=tc_obs_size,
      local_devices_to_use=local_devices_to_use,
      tcrmdp_network=tcrmdp_network,
      agent_policy_optimizer=agent_policy_optimizer,
      agent_q_optimizer=agent_q_optimizer,
      adversary_policy_optimizer=adversary_policy_optimizer,
      adversary_q_optimizer=adversary_q_optimizer,
      num_envs=num_envs,
      std_max=std_max,
      std_min=std_min,
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
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
    )
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap((training_state.normalizer_params, training_state.agent_policy_params)),
        training_metrics={},
    )
    progress_fn(0, metrics)

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

  training_walltime = 0.0
  current_step = 0
  for _ in range(num_evals_after_init):
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    training_state, env_state, buffer_state, training_metrics = (
        training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
    )
    current_step = int(_unpmap(training_state.env_steps))
    if process_id == 0:
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.agent_policy_params)
          ),
          training_metrics,
      )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  params = _unpmap(
      (training_state.normalizer_params, training_state.agent_policy_params)
  )
  logging.info("total steps: %s", total_steps)
  pmap.synchronize_hosts()
  return make_policy, params, metrics
