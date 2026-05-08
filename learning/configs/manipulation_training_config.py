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

# from mujoco_playground._src import dm_control_suite, locomotion
from custom_envs import manipulation


def _fast_td3_env_steps(num_iterations: int, num_envs: int) -> int:
  """Maps FastTD3 loop-count budgets to our env-step accounting."""
  return num_iterations * num_envs


def _apply_fast_td3_replay_defaults(rl_config: config_dict.ConfigDict):
  """Aligns replay sizing with FastTD3's per-environment circular buffer."""
  rl_config.min_replay_size = rl_config.num_envs * 10
  rl_config.max_replay_size = rl_config.num_envs * (1024 * 10)
  return rl_config

def manipulation_td3_config(env_name: str) -> config_dict.ConfigDict:
  """Returns TD3 config aligned with FastTD3 manipulation defaults."""
  
  env_config = manipulation.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=_fast_td3_env_steps(150_000, 1024),
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.97,
      learning_rate=3e-4,
      num_envs=1024,
      batch_size=32768,
      grad_updates_per_step=2,
      std_min=0.001,
      std_max=0.4,
      tau=0.1,
      policy_noise=0.001,
      noise_clip=0.5,
      policy_frequency=2,
      distributional_q=True,
      network_factory=config_dict.create(
          hidden_layer_sizes=(512, 256, 128),
          num_atoms=101,
          v_min=-10.0,
          v_max=10.0,
          policy_obs_key="state",
          value_obs_key="privileged_state",
      ),
  )
  rl_config = _apply_fast_td3_replay_defaults(rl_config)

  if env_name == "LeapCubeReorient":
    rl_config.discounting = 0.99
    rl_config.policy_noise = 0.2
    rl_config.network_factory = config_dict.create(
        hidden_layer_sizes=(512, 256, 128),
        num_atoms=101,
        v_min=-50.0,
        v_max=50.0,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  elif env_name == "LeapCubeRotateZAxis":
    rl_config.discounting = 0.99
    rl_config.policy_noise = 0.2
    rl_config.network_factory = config_dict.create(
        hidden_layer_sizes=(512, 256, 128),
        num_atoms=101,
        v_min=-10.0,
        v_max=10.0,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  rl_config = _apply_fast_td3_replay_defaults(rl_config)
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
  rl_config.rarl_range_scale = 0.5
  rl_config.network_factory = config_dict.create(
      hidden_layer_sizes=(256, 256),
      policy_network_layer_norm=False,
      q_network_layer_norm=False,
  )
  return rl_config


def manipulation_tcrmdp_config(env_name: str) -> config_dict.ConfigDict:
  """Returns TCRMDP-style config on top of MuJoCo Playground scale."""
  return _apply_tcrmdp_td3_defaults(manipulation_td3_config(env_name))
