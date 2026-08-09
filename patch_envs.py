#!/usr/bin/env python
"""
네 개 벤치마크 환경(특히 Walker, Acrobot)에 도메인 랜덤화를 활성화하는 통합 패치.

repo 루트(= custom_envs/ 와 learning/ 이 보이는 폴더)에서 실행:
    python patch_envs.py

하는 일 (전부 아래 코드에서 확인 가능):
  1) custom_envs/dm_control_suite/acrobot.py 를 새로 생성
       - mujoco_playground Acrobot 을 상속해 dr_range(논문 Table 9 범위) 추가
       - domain_randomize / _eval (위팔/아래팔 질량) 추가
       - obs 를 {state, privileged_state} dict 로 반환 (asymmetric critic 용)
  2) custom_envs/dm_control_suite/__init__.py 패치
       - Acrobot import/env/config 주석 해제
       - _randomizer / _randomizer_eval 에 WalkerWalk, AcrobotSwingup 등록
  3) learning/agents/td3/train.py 패치
       - 2D 환경에서만 실행되며 target_lnpdf(주석 처리된 필드)를 참조해 깨지는
         진단용 등고선 플롯 2곳을 비활성화 (학습/지표와 무관)
  4) learning/configs/dm_control_training_config.py 패치
       - asymmetric-critic network_factory 를 받는 벤치마크 리스트(sac/td3)에
         AcrobotSwingup, WalkerWalk 추가 (논문의 asymmetric critic 설계와 정합)

이미 적용된 항목은 건너뜁니다 (여러 번 실행해도 안전).
"""
import os
import sys

if not (os.path.isdir("custom_envs") and os.path.isdir("learning")):
    sys.exit("ERROR: repo 루트(custom_envs/ 와 learning/ 이 보이는 곳)에서 실행하세요.")


ACROBOT_PY = r'''
"""AcrobotSwingup with domain randomization over the two link masses.

Mirrors the structure of cheetah.py / walker.py: a thin subclass of the
mujoco_playground Acrobot env that exposes a `dr_range` property, plus
module-level `domain_randomize` / `domain_randomize_eval` functions.

Randomization ranges follow the paper (Table 9, AcrobotSwingup, 2D):
    Upper Arm Mass : [0.78, 3.0]
    Lower Arm Mass : [0.5, 1.2]
Both are absolute masses (the base model has each link mass = 1.0).
"""
import functools

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src.dm_control_suite import acrobot as _acrobot

# re-export the base task config unchanged
default_config = _acrobot.default_config

# body indices in the acrobot model (verified: body[1]=upper_arm, body[2]=lower_arm)
UPPER_ARM_BODY_ID = 1
LOWER_ARM_BODY_ID = 2


class Balance(_acrobot.Balance):
  """Acrobot Balance task that additionally advertises a DR range.

  Also exposes the observation as a {"state", "privileged_state"} dict so the
  asymmetric critic works, mirroring the other custom DM-Control envs
  (cartpole/cheetah/walker). The DR-augmented critic appends the dynamics
  parameters to the privileged branch downstream.
  """

  def _get_obs(self, data, info):
    state = super()._get_obs(data, info)
    if isinstance(state, dict):
      return state
    return {"state": state, "privileged_state": state}

  @property
  def dr_range(self):
    low = jp.array([
        0.78,  # upper arm mass
        0.5,   # lower arm mass
    ])
    high = jp.array([
        3.0,   # upper arm mass
        1.2,   # lower arm mass
    ])
    return low, high


def domain_randomize(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  """Randomize the two link masses (training path; vmapped internally)."""
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(
        jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)

  @jax.vmap
  def shift_dynamics(params):
    body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(params[0])
    body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(params[1])
    return (body_mass,)

  @jax.vmap
  def rand_dynamics(rng):
    p = dist(rng)
    body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(p[0])
    body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(p[1])
    return (body_mass,)

  if rng is None and params is not None:
    (body_mass,) = shift_dynamics(params)
  elif rng is not None and params is None:
    (body_mass,) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"body_mass": 0})
  model = model.tree_replace({"body_mass": body_mass})
  return model, in_axes


def domain_randomize_eval(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  """Randomize the two link masses (eval path; not vmapped internally)."""
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(
        jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)

  def shift_dynamics(params):
    body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(params[0])
    body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(params[1])
    return (body_mass,)

  def rand_dynamics(rng):
    p = dist(rng)
    body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(p[0])
    body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(p[1])
    return (body_mass,)

  if rng is None and params is not None:
    (body_mass,) = shift_dynamics(params)
  elif rng is not None and params is None:
    (body_mass,) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"body_mass": 0})
  model = model.tree_replace({"body_mass": body_mass})
  return model, in_axes
'''


def write_file(path, content):
    if os.path.isfile(path) and open(path).read() == content:
        print(f"[skip] {path}: 이미 동일")
        return
    with open(path, "w") as f:
        f.write(content)
    print(f"[ok]   {path}: 생성/갱신")


def patch(path, edits):
    s = open(path).read()
    for label, old, new in edits:
        if new in s:
            print(f"[skip] {os.path.basename(path)} - {label}: 이미 적용됨")
            continue
        if old not in s:
            sys.exit(f"[ERROR] {os.path.basename(path)} - {label}: 기준 문자열을 못 찾음")
        s = s.replace(old, new, 1)
        print(f"[ok]   {os.path.basename(path)} - {label}: 적용")
    open(path, "w").write(s)


# 1) acrobot.py 생성
write_file(os.path.join("custom_envs", "dm_control_suite", "acrobot.py"), ACROBOT_PY)

# 2) __init__.py 패치
INIT = os.path.join("custom_envs", "dm_control_suite", "__init__.py")
patch(INIT, [
    ("Acrobot import",
     "# from mujoco_playground._src.dm_control_suite import acrobot",
     "from custom_envs.dm_control_suite import acrobot"),
    ("Acrobot env 등록",
     '    # "AcrobotSwingup": partial(acrobot.Balance, sparse=False),',
     '    "AcrobotSwingup": partial(acrobot.Balance, sparse=False),'),
    ("Acrobot config 등록",
     '    # "AcrobotSwingup": acrobot.default_config,',
     '    "AcrobotSwingup": acrobot.default_config,'),
    ("randomizer(학습) 등록",
     '  "CheetahRun" : cheetah.domain_randomize,\n}',
     '  "CheetahRun" : cheetah.domain_randomize,\n'
     '  "WalkerWalk" : walker.domain_randomize,\n'
     '  "AcrobotSwingup" : acrobot.domain_randomize,\n}'),
    ("randomizer(평가) 등록",
     '  "CheetahRun" : cheetah.domain_randomize_eval,\n}',
     '  "CheetahRun" : cheetah.domain_randomize_eval,\n'
     '  "WalkerWalk" : walker.domain_randomize_eval,\n'
     '  "AcrobotSwingup" : acrobot.domain_randomize_eval,\n}'),
])

# 3) td3/train.py 패치 (깨진 2D 진단 플롯 비활성화)
TD3 = os.path.join("learning", "agents", "td3", "train.py")
_note = "  # [disabled] 2D occupancy debug plot: references target_lnpdf (N/A for UDR td3)"
patch(TD3, [
    ("2D 플롯 게이트 1", "  if  len(dr_range_low)==2:", "  if  False:" + _note),
    ("2D 플롯 게이트 2", "    if len(dr_range_low)==2:", "    if False:" + _note),
])

# 4) config 패치 (벤치마크 리스트에 Acrobot, WalkerWalk 추가)
CFG = os.path.join("learning", "configs", "dm_control_training_config.py")
patch(CFG, [
    ("sac 벤치마크 리스트",
     '  if env_name in ("CheetahRun","WalkerRun", "PendulumSwingUp", "HumanoidWalk", "CartpoleSwingup"):',
     '  if env_name in ("CheetahRun","WalkerRun", "PendulumSwingUp", "HumanoidWalk", "CartpoleSwingup", "AcrobotSwingup", "WalkerWalk"):'),
    ("td3 벤치마크 리스트",
     '  if env_name in ("CheetahRun","WalkerRun", "PendulumSwingUp", "HumanoidWalk", "CartpoleSwingup","HopperHop"):',
     '  if env_name in ("CheetahRun","WalkerRun", "PendulumSwingUp", "HumanoidWalk", "CartpoleSwingup","HopperHop", "AcrobotSwingup", "WalkerWalk"):'),
])

print("\n완료: 4개 환경 도메인 랜덤화 활성화 패치가 적용되었습니다.")
