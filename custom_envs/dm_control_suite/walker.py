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
"""Walker environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from custom_envs import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common
import functools
_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "walker.xml"
# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
WALK_SPEED = 1
RUN_SPEED = 8


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.025,
      sim_dt=0.0025,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      naconmax=50_000,
      njmax=400,
  )


class PlanarWalker(mjx_env.MjxEnv):
  """A planar walker task."""

  def __init__(
      self,
      move_speed: float,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._move_speed = move_speed
    if self._move_speed == 0.0:
      self._get_reward = self._get_stand_reward
    else:
      self._get_reward = self._get_move_reward

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model =  mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._torso_id = self.mj_model.body("torso").id
    self._lowers = self._mj_model.jnt_range[3:, 0]
    self._uppers = self._mj_model.jnt_range[3:, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng2 = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[2].set(
        jax.random.uniform(rng1, (), minval=-jp.pi, maxval=jp.pi)
    )
    qpos = qpos.at[3:].set(
        jax.random.uniform(
            rng2,
            (self.mjx_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
        )
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos)

    metrics = {
        "reward/standing": jp.zeros(()),
        "reward/upright": jp.zeros(()),
        "reward/stand": jp.zeros(()),
        "reward/move": jp.zeros(()),
    }
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
    orientations = data.xmat[1:, [0, 0], [0, 2]].ravel()
    height = data.xmat[self._torso_id, 2, 2]
    velocity = data.qvel
    return jp.concatenate([
        orientations,
        height.reshape(1),
        velocity,
    ])

  def _get_stand_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info  # Unused.

    torso_height = data.xpos[self._torso_id, -1]
    standing = reward.tolerance(
        torso_height,
        bounds=(_STAND_HEIGHT, float("inf")),
        margin=_STAND_HEIGHT / 2,
    )
    metrics["reward/standing"] = standing

    torso_upright = data.xmat[self._torso_id, 2, 2]  # zz component.
    upright = (1 + torso_upright) / 2
    metrics["reward/upright"] = upright

    stand_reward = (3 * standing + upright) / 4
    metrics["reward/stand"] = stand_reward

    return stand_reward

  def _get_move_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    stand_reward = self._get_stand_reward(data, action, info, metrics)

    horizontal_velocity = mjx_env.get_sensor_data(
        self.mj_model, data, "torso_subtreelinvel"
    )[0]
    move_reward = reward.tolerance(
        horizontal_velocity,
        bounds=(self._move_speed, float("inf")),
        margin=self._move_speed / 2,
        value_at_margin=0.5,
        sigmoid="linear",
    )
    metrics["reward/move"] = move_reward

    return stand_reward * (5 * move_reward + 1) / 6

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
    # 논문 Table 9 WalkerWalk(15D): 순서 = 마찰, 몸통, 허벅지(L,R), 종아리(L,R), 발(L,R), body offset 7D
    low = jp.array([
        0.82,                        # floor friction
        0.88,                        # torso mass
        0.98, 0.98,                  # thigh mass (Left, Right)
        0.68, 0.68,                  # leg mass   (Left, Right)
        0.74, 0.74,                  # foot mass  (Left, Right)
        -0.05, -0.05, -0.05, -0.05,  # body ipos offset (7D) : body 1~7
        -0.05, -0.05, -0.05,
    ])
    high = jp.array([
        7.58,                        # floor friction
        6.19,                        # torso mass
        6.87, 6.87,                  # thigh mass (Left, Right)
        4.75, 4.75,                  # leg mass   (Left, Right)
        5.15, 5.15,                  # foot mass  (Left, Right)
        0.05, 0.05, 0.05, 0.05,      # body ipos offset (7D)
        0.05, 0.05, 0.05,
    ])
    return low, high
  
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
THIGH_BODY_ID = 2

FLOOR_GEOM_ID = 0
# body id: 1=torso, 2=right_thigh, 3=right_leg, 4=right_foot, 5=left_thigh, 6=left_leg, 7=left_foot
TORSO_ID = 1
RIGHT_THIGH_ID, RIGHT_LEG_ID, RIGHT_FOOT_ID = 2, 3, 4
LEFT_THIGH_ID, LEFT_LEG_ID, LEFT_FOOT_ID = 5, 6, 7


def _walker_fields(model, p):
  """p로부터 랜덤화된 (마찰, 질량, ipos오프셋) 필드를 만든다.
  p[0]=마찰, p[1]=몸통, p[2:4]=허벅지(L,R), p[4:6]=종아리(L,R), p[6:8]=발(L,R),
  p[8:15]=body 1~7의 ipos z축 오프셋(7D).
  주의: 논문의 'Body offset 7D'가 어느 축인지 명시가 없어 z축(index 2)에 적용한다고 가정함.
        (논문 공개코드가 있으면 축을 맞춰 확인 필요)"""
  geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(p[0])
  body_mass = model.body_mass
  body_mass = body_mass.at[TORSO_ID].set(p[1])
  body_mass = body_mass.at[LEFT_THIGH_ID].set(p[2])
  body_mass = body_mass.at[RIGHT_THIGH_ID].set(p[3])
  body_mass = body_mass.at[LEFT_LEG_ID].set(p[4])
  body_mass = body_mass.at[RIGHT_LEG_ID].set(p[5])
  body_mass = body_mass.at[LEFT_FOOT_ID].set(p[6])
  body_mass = body_mass.at[RIGHT_FOOT_ID].set(p[7])
  # body 1~7 의 ipos z성분에 per-body 오프셋을 더함 (7D)
  body_ipos = model.body_ipos.at[1:8, 2].add(p[8:15])
  return geom_friction, body_mass, body_ipos


def domain_randomize(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  """학습용: rng면 균등분포 샘플, params면 그 값 사용. (내부 vmap)"""
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)

  @jax.vmap
  def shift_dynamics(params):
    return _walker_fields(model, params)

  @jax.vmap
  def rand_dynamics(rng):
    return _walker_fields(model, dist(rng))

  if rng is None and params is not None:
    geom_friction, body_mass, body_ipos = shift_dynamics(params)
  elif rng is not None and params is None:
    geom_friction, body_mass, body_ipos = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0, "body_ipos": 0})
  model = model.tree_replace({"geom_friction": geom_friction, "body_mass": body_mass, "body_ipos": body_ipos})
  return model, in_axes


def domain_randomize_eval(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  """평가용: 내부 vmap 없음."""
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)

  def shift_dynamics(params):
    return _walker_fields(model, params)

  def rand_dynamics(rng):
    return _walker_fields(model, dist(rng))

  if rng is None and params is not None:
    geom_friction, body_mass, body_ipos = shift_dynamics(params)
  elif rng is not None and params is None:
    geom_friction, body_mass, body_ipos = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0, "body_ipos": 0})
  model = model.tree_replace({"geom_friction": geom_friction, "body_mass": body_mass, "body_ipos": body_ipos})
  return model, in_axes
