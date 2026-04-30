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

"""WDTD3 training with multi-GPU (pmap) support."""

from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import functools
import os
import time

import psutil
from absl import logging
from brax import base, envs
from brax.training import acting, gradients, pmap, replay_buffers, types
from brax.training.acme import running_statistics, specs
from brax.training.types import Params, PRNGKey
from brax.training.acme.types import NestedArray
from brax.envs.base import Env
import flax
import jax
import jax.numpy as jnp
import optax
import copy
# If your queue is not pmap-friendly, replace with UniformSamplingQueue.
from module.buffer import DynamicBatchQueue  # noqa: E402

from agents.wdtd3 import checkpoint
from agents.wdtd3 import losses as wdtd3_losses
from agents.wdtd3 import networks as wdtd3_networks
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.drhard_wrapper import wrap_for_hard_dr_training
from learning.module.wrapper.wrapper import Wrapper
from learning.module.wrapper.evaluator import AdvEvaluator, Evaluator

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

_PMAP_AXIS_NAME = 'i'

class DRTransition(NamedTuple):
  observation: NestedArray
  action: NestedArray
  reward: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  extras: NestedArray = ()


ReplayBufferState = Any


@flax.struct.dataclass
class TrainingState:
  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  gradient_steps: types.UInt64
  env_steps: types.UInt64
  lmbda_optimizer_state: optax.OptState
  lmbda_params: Params
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


def _init_training_state(
    key: PRNGKey,
    obs_size: Union[int, Dict[str, specs.Array]],
    local_devices_to_use: int,
    wdtd3_network: wdtd3_networks.WDTD3Networks,
    policy_optimizer: optax.GradientTransformation,
    lmbda_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    single_lambda: bool,
    per_replica_batch: int,
    init_lmbda: float,
    num_envs : int,
    num_nominals : int,
    std_max: float =0.4,
    std_min : float =0.05,
) -> TrainingState:
  """Init + replicate TrainingState across devices."""
  key_policy, key_q, key_noise = jax.random.split(key, 3)

  if single_lambda:
    lmbda_params = jnp.asarray(init_lmbda, dtype=jnp.float32)
  else:
    # λ per-sample in a replica's local minibatch (before grad_updates_per_step split)
    lmbda_params = init_lmbda * jnp.ones((per_replica_batch,), dtype=jnp.float32)
  lmbda_optimizer_state = lmbda_optimizer.init(lmbda_params)

  policy_params = wdtd3_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)

  q_params = wdtd3_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  normalizer_params = running_statistics.init_state(obs_size)

  ts = TrainingState(
      policy_optimizer_state=policy_optimizer_state,
      policy_params=policy_params,
      q_optimizer_state=q_optimizer_state,
      q_params=q_params,
      target_q_params=q_params,
      gradient_steps=types.UInt64(hi=0, lo=0),
      env_steps=types.UInt64(hi=0, lo=0),
      lmbda_optimizer_state=lmbda_optimizer_state,
      lmbda_params=lmbda_params,
      normalizer_params=normalizer_params,
      noise_scales= jax.random.normal(key_noise,\
         (num_envs // jax.process_count() // local_devices_to_use, )) *(std_max - std_min) + std_min,
  )
  return _replicate_across_devices(ts, local_devices_to_use)


def train(
    environment: Env,
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
    max_devices_per_host: Optional[int] = 4,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    n_nominals: int = 10,
    delta: float = 0.1,
    lambda_update_steps: int = 100,
    single_lambda: bool = False,
    distance_type: str = "wass",
    lmbda_lr: float = 3e-4,
    init_lmbda: float = 0.0,
    network_factory: types.NetworkFactory[
        wdtd3_networks.WDTD3Networks
    ] = wdtd3_networks.make_wdtd3_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    dr_train_ratio: float = 1.0,
    std_max=0.4,
    std_min=0.05,
    policy_noise=0.2,
    noise_clip=0.5,
):
  """WDTD3 training (multi-GPU)."""
  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  logging.info('local_device_count: %s; total_device_count: %s',
               local_devices_to_use, device_count)

  if min_replay_size >= num_timesteps:
    raise ValueError('No training will happen because min_replay_size >= num_timesteps')

  if max_replay_size is None:
    max_replay_size = num_timesteps

  if num_envs % device_count != 0:
    raise ValueError(f'num_envs ({num_envs}) must be divisible by device_count ({device_count})')
  num_envs_per_device = num_envs // device_count

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

  rng = jax.random.PRNGKey(seed)
  rng, key = jax.random.split(rng)

  if hasattr(env,'dr_range') :
    dr_low, dr_high = env.dr_range
    dr_mid = (dr_low + dr_high) / 2.
    dr_scale = (dr_high - dr_low) / 2.
    training_dr_range = (dr_mid - dr_train_ratio*dr_scale, dr_mid + dr_train_ratio*dr_scale)
  else:
    training_dr_range = env.dr_range
  # Per device randomization keys
  v_randomization_fn = functools.partial(
      randomization_fn,
      dr_range=training_dr_range
  )

  # Important: n_envs is PER-DEVICE here
  env = wrap_for_hard_dr_training(
      env,
      n_nominals=n_nominals,
      n_envs=num_envs_per_device,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )

  action_size = env.action_size

  normalize_fn = lambda x, y: x 
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  wdtd3_network = network_factory(
      observation_size=env.observation_size,
      action_size=action_size,
      preprocess_observations_fn=normalize_fn,
  )
  make_policy = wdtd3_networks.make_inference_fn(wdtd3_network)
  lmbda_optimizer = optax.adam(learning_rate=lmbda_lr)
  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)

  # Dummy samples for buffer construction (dict obs supported)
  obs_size = env.observation_size
  dummy_obs = {k: jnp.zeros(obs_size[k]) for k in obs_size} if isinstance(obs_size, dict) else jnp.zeros((obs_size,))
  dummy_action = jnp.zeros((action_size,))
  dummy_next_obs = (
      {k: jnp.zeros((n_nominals,) + obs_size[k]) for k in obs_size}
      if isinstance(obs_size, dict) else jnp.zeros((n_nominals, obs_size))
  )
  dummy_transition = DRTransition(
      observation=dummy_obs,
      action=dummy_action,
      reward=jnp.zeros((n_nominals,)),
      discount=jnp.zeros((n_nominals,)),
      next_observation=dummy_next_obs,
      extras={'state_extras': {'truncation': jnp.zeros((n_nominals,))}, 'policy_extras': {}},
  )

  # Per-replica sample size
  per_replica_sample = batch_size * grad_updates_per_step // device_count

  replay_buffer = DynamicBatchQueue(  # if not pmap-safe, use UniformSamplingQueue
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=per_replica_sample,
  )

  # Loss fns
  lmbda_loss, kl_lmbda_loss, tv_lmbda_loss, critic_loss, actor_loss = wdtd3_losses.make_losses(
      wdtd3_network=wdtd3_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
  )

  # Gradient update fns aggregated across devices
  if distance_type == "kl":
    lmbda_update = gradients.gradient_update_fn(kl_lmbda_loss, lmbda_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME)
  elif distance_type == "tv":
    lmbda_update = gradients.gradient_update_fn(tv_lmbda_loss, lmbda_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME)
  else:
    lmbda_update = gradients.gradient_update_fn(lmbda_loss, lmbda_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME)

  critic_update = gradients.gradient_update_fn(critic_loss, q_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME)
  actor_update  = gradients.gradient_update_fn(actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
  
  def sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: DRTransition
    ):
    training_state, key = carry
    key, key_lmbda, key_critic, key_actor, key_noise = jax.random.split(key, 5)
    noise = jax.random.normal(key_noise, shape=(transitions.action.shape[0], n_nominals, transitions.action.shape[-1])) * policy_noise
    noise = jnp.clip(noise,-noise_clip, noise_clip)
    # Inner λ optimization via scan
    def lambda_optimization_step(carry, _):
      lmbda_params, lmbda_opt_state, prev_loss = carry
      (lmbda_loss_val, (next_v, min_indices, loss_info)), new_lmbda_params, new_lmbda_opt_state = lmbda_update(
          lmbda_params,
          training_state.policy_params,
          training_state.normalizer_params,
          training_state.target_q_params,
          transitions,
          n_nominals,
          noise,
          delta,
          prev_loss,
          key_lmbda,
          optimizer_state=lmbda_opt_state,
      )
      return (new_lmbda_params, new_lmbda_opt_state, lmbda_loss_val), (lmbda_loss_val, next_v, min_indices, loss_info)

    (lmbda_params, lmbda_opt_state, _), (lmbda_losses, next_vs, min_idcs, loss_infos) = jax.lax.scan(
        lambda_optimization_step,
        (training_state.lmbda_params, training_state.lmbda_optimizer_state, jnp.array(0.0, dtype=jnp.float32)),
        xs=None,
        length=lambda_update_steps,
    )
    lmbda_loss = lmbda_losses[-1]
    next_v = next_vs[-1]
    min_indices = min_idcs[-1]
    loss_info = loss_infos[-1]

    # Critic update (consumes next_v/min_indices computed under robust backup)
    (critic_loss, (current_q, _)), q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.normalizer_params,
        transitions,
        next_v,
        min_indices,
        optimizer_state=training_state.q_optimizer_state,
    )

    # Actor update
    actor_loss, policy_params, policy_opt_state = actor_update(
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.q_params,
        transitions,
        key_actor,
        optimizer_state=training_state.policy_optimizer_state,
    )

    # Target soft update
    new_target_q_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_q_params,
        q_params,
    )

    metrics = {
        'lmbda_loss': lmbda_loss,
        'lmbda_loss_reduction': loss_info,
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'lmbda': lmbda_params,
        'current_q_min' : current_q.min(),
        'current_q_max' : current_q.max(),
        'current_q_mean' : current_q.mean(),
        'next_v_min' : next_v.min(),
        'next_v_max' : next_v.max(),
        'next_v_mean' : next_v.mean(),
    }

    new_ts = TrainingState(
        policy_optimizer_state=policy_opt_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        lmbda_optimizer_state=lmbda_opt_state,
        lmbda_params=lmbda_params,
        normalizer_params=training_state.normalizer_params,
        noise_scales=training_state.noise_scales,
    )
    return (new_ts, key), metrics

  def random_step(
    env: envs.Env,
    env_state: envs.State,
    policy: types.Policy,
    noise_scales: jnp.ndarray,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
  ):
    env_key, action_key, rollout_key = jax.random.split(key,3)
    actions, policy_extras = policy(env_state.obs, noise_scales, action_key)
    nstate = env.step(env_state, actions, env_key)
    nstate = jax.tree_util.tree_map(lambda x : x.reshape((num_envs_per_device, n_nominals) + x.shape[1:])  , nstate)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    idx = jax.random.choice(rollout_key, a = jnp.arange(n_nominals), shape=(num_envs_per_device,))
    rollout_next_state = jax.tree_util.tree_map(lambda x : x[jnp.arange(num_envs_per_device), idx], nstate)

    return rollout_next_state, DRTransition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation= nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
  )

  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      noise_scales : jnp.ndarray,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    noise_key, key = jax.random.split(key)
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = random_step(env, env_state, policy, noise_scales, key, extra_fields=('truncation',))

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    noise_scales = (1-env_state.done)*noise_scales + \
          env_state.done* (jax.random.normal(noise_key, shape=noise_scales.shape) *(std_max - std_min) + std_min)
    simul_info ={
      "simul/reward_mean" : transitions.reward.mean(),
      "simul/reward_std" : transitions.reward.std(),
      "simul/reward_max" : transitions.reward.max(),
      "simul/reward_min" : transitions.reward.min(),
      # "simul/dynamics_params_mean" : dynamics_params.mean(),
      # "simul/dynamics_params_std" : dynamics_params.std(),
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
    normalizer_params, noise_scales, env_state, buffer_state, simul_info = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.noise_scales,
        env_state,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        noise_scales = noise_scales,
        env_steps=training_state.env_steps + env_steps_per_actor_step, #// device_count,  # per replica
    )

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    # reshape front dim to [grad_updates_per_step, per_replica_batch, ...]
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
    metrics.update(simul_info)
    return training_state, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    def f(carry, _):
      ts, es, bs, k = carry
      k, new_k = jax.random.split(k)
      new_norm_params, new_noise_scales, es, bs, simul_info = get_experience(
          ts.normalizer_params,
          ts.policy_params,
          ts.noise_scales,
          es,
          bs,
          k,
      )
      ts = ts.replace(
          normalizer_params=new_norm_params,
          noise_scales=new_noise_scales,
          env_steps=ts.env_steps + env_steps_per_actor_step,# // device_count,  # per replica
      )
      return (ts, es, bs, new_k), ()
    return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]

  # === pmap wrappers ===
  prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)
  
  def training_epoch(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

    def f(carry, unused_t):
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

  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ):
    nonlocal training_walltime
    t0 = time.time()
    training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, key)
    # metrics is a pytree of arrays per step; average along scan dim then across devices
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
    epoch_time = time.time() - t0
    training_walltime += epoch_time
    sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{k}': v for k, v in metrics.items()},
    }
    return training_state, env_state, buffer_state, metrics

  # === Keys & env init ===
  rng = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(rng)
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

  # env.reset per device
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
  print("num envs", num_envs)
  print("n_nominals", n_nominals)
  print(" env keys shape", env_keys.shape)
  env_state = jax.pmap(env.reset)(env_keys)

  # Build obs specs per device for normalizer init
  obs_specs = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )

  # === Training state init (replicated) ===
  per_replica_batch_nominal = per_replica_sample // grad_updates_per_step
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_specs,
      local_devices_to_use=local_devices_to_use,
      wdtd3_network=wdtd3_network,
      policy_optimizer=policy_optimizer,
      lmbda_optimizer=lmbda_optimizer,
      q_optimizer=q_optimizer,
      single_lambda=single_lambda,
      per_replica_batch=per_replica_batch_nominal,
      init_lmbda=init_lmbda,
      num_envs=num_envs,
      num_nominals=n_nominals,
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
    )

  # === Replay buffer init (per device) ===
  buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

  # === Evaluator ===
  eval_env = copy.deepcopy(environment)
  eval_dr_low, eval_dr_high = environment.dr_range
  eval_env = wrap_for_adv_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=functools.partial(
          randomization_fn,
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

  # Optional initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap((training_state.normalizer_params, training_state.policy_params)),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # === Prefill ===
  process = psutil.Process(os.getpid())
  t0 = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_keys
  )

  replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size
  training_walltime = time.time() - t0

  # === Main loop ===
  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    training_state, env_state, buffer_state, training_metrics = training_epoch_with_timing(
        training_state, env_state, buffer_state, epoch_keys
    )
    current_step = int(_unpmap(training_state.env_steps))

    if process_id == 0:
      if checkpoint_logdir:
        params = _unpmap((training_state.normalizer_params, training_state.policy_params))
        ckpt_config = checkpoint.network_config(
            observation_size=obs_specs,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)

      metrics = evaluator.run_evaluation(
          _unpmap((training_state.normalizer_params, training_state.policy_params)),
          training_metrics,
      )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if total_steps < num_timesteps:
    raise AssertionError(f'Total steps {total_steps} is less than num_timesteps {num_timesteps}.')

  params = _unpmap((training_state.normalizer_params, training_state.policy_params))
  pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
