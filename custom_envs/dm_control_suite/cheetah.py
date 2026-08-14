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
"""Cheetah environment."""

import functools
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from custom_envs import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common
from omegaconf import OmegaConf

ROOT_PATH = Path(__file__).parent
_XML_PATH = ROOT_PATH / "xmls" / "cheetah.xml"
# Running speed above which reward is 1.
_RUN_SPEED = 10


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      naconmax=100_000,
      njmax=100,
  )


class Run(mjx_env.MjxEnv):
  """Cheetah running environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._lowers = self._mj_model.jnt_range[3:, 0]
    self._uppers = self._mj_model.jnt_range[3:, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1 = jax.random.split(rng, 2)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[3:].set(
        jax.random.uniform(
            rng1,
            (self.mjx_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
        )
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Stabilize.
    data = mjx_env.step(self.mjx_model, data, jp.zeros(self.mjx_model.nu), 200)
    data = data.replace(time=0.0)

    metrics = {}
    info = {"rng": rng}

    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    state = jp.concatenate([
        data.qpos[1:],
        data.qvel,
    ])
    privileged_state = state
    # return state
    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info, metrics  # Unused.
    speed = mjx_env.get_sensor_data(self.mj_model, data, "torso_subtreelinvel")[
        0
    ]  # x-axis only.
    return reward.tolerance(
        speed,
        bounds=(_RUN_SPEED, float("inf")),
        margin=_RUN_SPEED,
        value_at_margin=0,
        sigmoid="linear",
    )

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
  
  @property
  def dr_range(self) -> dict:
    # 논문 Table 9 CheetahRun(2D): Floor Friction [0.01,3.5], Torso Mass [0.1,23.0]
    low = jp.array([
        0.01,  # floor friction
        0.1,   # torso mass
    ])
    high = jp.array([
        3.5,   # floor friction
        23.0,  # torso mass
    ])
    return low, high

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
BTHIGH_BODY_ID = 2

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def _cheetah_fields(model, p):
  """파라미터 벡터 p로부터 랜덤화된 필드들을 만든다. p[0]=마찰, p[1]=몸통질량."""
  geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(p[0])
  body_mass = model.body_mass.at[TORSO_BODY_ID].set(p[1])
  return geom_friction, body_mass


def domain_randomize(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  """학습용: rng가 오면 균등분포에서 뽑고, params가 오면 그 값을 그대로 사용. (내부 vmap)"""
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)

  @jax.vmap
  def shift_dynamics(params):
    return _cheetah_fields(model, params)

  @jax.vmap
  def rand_dynamics(rng):
    return _cheetah_fields(model, dist(rng))

  if rng is None and params is not None:
    geom_friction, body_mass = shift_dynamics(params)
  elif rng is not None and params is None:
    geom_friction, body_mass = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0})
  model = model.tree_replace({"geom_friction": geom_friction, "body_mass": body_mass})
  return model, in_axes


def domain_randomize_eval(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  """평가용: 학습용과 동일하나 내부 vmap을 쓰지 않음(호출부에서 vmap)."""
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)

  def shift_dynamics(params):
    return _cheetah_fields(model, params)

  def rand_dynamics(rng):
    return _cheetah_fields(model, dist(rng))

  if rng is None and params is not None:
    geom_friction, body_mass = shift_dynamics(params)
  elif rng is not None and params is None:
    geom_friction, body_mass = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0})
  model = model.tree_replace({"geom_friction": geom_friction, "body_mass": body_mass})
  return model, in_axes
