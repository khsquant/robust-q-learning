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
