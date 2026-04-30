# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""RL config for DM Control Suite."""

from ml_collections import config_dict

from mujoco_playground._src import dm_control_suite


def brax_sac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      num_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 10_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 30_000_000
  if env_name in ("CheetahRun","WalkerRun", "PendulumSwingUp", "HumanoidWalk", "CartpoleSwingup"):
    rl_config.network_factory = config_dict.create(
      q_network_layer_norm=True,
      policy_obs_key="state",
      value_obs_key="privileged_state",
    )
  return rl_config
def brax_td3_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      num_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      std_min=0.01,
      std_max=0.4,
      policy_noise=0.2,
      noise_clip=0.5,
      policy_frequency=2,
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 10_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    std_min=0.1
    rl_config.num_timesteps = 30_000_000
  if env_name in ("CheetahRun","WalkerRun", "PendulumSwingUp", "HumanoidWalk", "CartpoleSwingup","HopperHop"):
    rl_config.network_factory = config_dict.create(
      q_network_layer_norm=True,
      distributional_q = False,
      v_min = -500.0,
      v_max = 500.0,
      policy_obs_key="state",
      value_obs_key="privileged_state",
    )
    rl_config.distributional_q = False
    rl_config.policy_noise= 0.1
    rl_config.std_min = 0.1
  return rl_config


def _apply_tcrmdp_td3_defaults(rl_config: config_dict.ConfigDict):
  """Applies TCRMDP TD3 defaults while preserving environment scale."""
  rl_config.reward_scaling = 1.0
  rl_config.normalize_observations = False
  rl_config.discounting = 0.99
  rl_config.learning_rate = 3e-4
  rl_config.batch_size = 256
  rl_config.grad_updates_per_step = 1
  rl_config.tau = 0.005
  rl_config.std_min = 0.1
  rl_config.std_max = 0.1
  rl_config.policy_noise = 0.2
  rl_config.noise_clip = 0.5
  rl_config.policy_frequency = 2
  rl_config.radius = 0.001
  rl_config.network_factory = config_dict.create(
      hidden_layer_sizes=(256, 256),
      policy_network_layer_norm=False,
      q_network_layer_norm=False,
  )
  return rl_config


def brax_tcrmdp_config(env_name: str) -> config_dict.ConfigDict:
  """Returns TCRMDP-style config on top of MuJoCo Playground scale."""
  return _apply_tcrmdp_td3_defaults(brax_td3_config(env_name))


def brax_wdsac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,  
      learning_rate=1e-3,
      num_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 8,
      min_replay_size=8192,

      network_factory=config_dict.create(
          q_network_layer_norm=True,
          policy_obs_key="state",
        value_obs_key="state",
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
  if env_name =="HopperHop":
    rl_config.num_timesteps = 10_000_000
  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 20_000_000
  if env_name == "CheetahRun":
    rl_config.network_factory = config_dict.create(
        q_network_layer_norm=True,
        policy_obs_key="state",
        value_obs_key="state",
    )
  return rl_config
