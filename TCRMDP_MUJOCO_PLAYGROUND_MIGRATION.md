# TCRMDP to MuJoCo Playground Migration Prompt

This file is the implementation prompt and debug contract for migrating the
TCRMDP algorithms into the `distributionally_robust_learning` MuJoCo Playground
project.  The target implementation must preserve the TCRMDP algorithmic
structure while fitting the existing JAX/Brax q-learning training stack.

## Source Algorithms

Use these TCRMDP files as the behavioral source of truth:

- `TCRMDP/src/main_oracle_tc_rarl.py`
- `TCRMDP/src/main_oracle_tc_m2td3.py`
- `TCRMDP/src/main_vanilla_tc_m2td3.py`
- `TCRMDP/src/tc_mdp.py`
- `TCRMDP/src/td3/trainer.py`
- `TCRMDP/src/td3/td3.py`

Do not copy the PyTorch/Gym implementation literally.  Recreate the same
algorithm roles in JAX:

- **Vanilla TC-M2TD3**: agent actor observes normal state, critic is omniscient
  through the current dynamics parameters `psi`, adversary changes parameters.
- **Oracle TC-RARL**: agent and adversary are separate TD3 learners; agent sees
  oracle state including normalized `psi`; adversary reward is `-reward`.
- **Oracle TC-M2TD3**: agent has an oracle/omniscient critic; adversary has only
  an actor trained through the shared agent critic/replay to minimize Q.

## MuJoCo Playground Mapping

TCRMDP mutates Gym environments with:

```text
env.set_params(new_params)
env.step(agent_action)
```

In this project, dynamics must be functional and batched:

```text
next_params = clip(current_params + radius * adversary_action, dr_low, dr_high)
next_state = env.step(state, agent_action, next_params)
```

Use `wrap_for_adv_training` as the only randomized training wrapper for these
algorithms.  It owns explicit perturbation parameters through
`state.info["dr_params"]`; do not introduce a second DR wrapper or hidden
environment mutation path.

Parameter normalization for oracle observations:

```text
psi = (params - dr_low) / (dr_high - dr_low)
```

Parameter reset rule:

- At environment reset, sample `dr_params` uniformly in the training DR range.
- During an episode, update params with the adversary action and `radius`.
- On done/reset, use the wrapper reset params as the new episode params.

## Observation and Action Contracts

Use dictionary observations internally so the existing policy/critic factories
can select keys.

- `state`: actor-visible observation.
- `critic_state`: critic-visible observation when it differs from actor state.
- `adv_state`: adversary-visible observation.

Construction rules:

- Base environment observation comes from `obs["state"]` when available.
- Existing privileged observation remains available and must not be removed.
- Vanilla TC-M2TD3:
  - agent actor input: base state only.
  - critic input: concat base state and normalized `psi`.
  - adversary input: concat base state, normalized `psi`, and current agent action.
- Oracle TC-RARL:
  - agent actor/critic input: concat base state and normalized `psi`.
  - adversary input: concat base state, normalized `psi`, and current agent action.
  - adversary reward: negative environment reward.
- Oracle TC-M2TD3:
  - agent actor input: concat base state and normalized `psi`.
  - critic input: concat base state and normalized `psi`.
  - adversary input: concat base state, normalized `psi`, and current agent action.
  - adversary actor update minimizes Q through the shared agent critic.

The adversary action dimension is `len(dr_low)`.  It must be clipped/tanh-bounded
to `[-1, 1]`.

## Hyperparameter Contract

Current MuJoCo Playground configs remain authoritative for scale:

- `num_timesteps`
- `num_envs`
- `min_replay_size`
- `max_replay_size`
- `episode_length`
- `action_repeat`
- `num_evals`

TCRMDP defaults are authoritative for TD3 mechanics:

- `learning_rate = 3e-4`
- `discounting = 0.99`
- `tau = 0.005`
- `reward_scaling = 1.0`
- `normalize_observations = False`
- `grad_updates_per_step = 1`
- `policy_noise = 0.2`
- `noise_clip = 0.5`
- `policy_frequency = 2`
- `std_min = std_max = 0.1` for exploration-noise equivalence
- `batch_size = 256`, unless a JAX device/local batch divisibility issue requires
  the smallest safe multiple that preserves the same intent.
- `radius = 0.001` for TC parameter changes.
- network hidden sizes `(256, 256)` for actors and critics.

Do not keep MuJoCo Playground TD3 learning-rate, discount, reward-scale,
normalization, batch-size, gradient-update, or architecture overrides for these
TC algorithms.  The point is to preserve current environment scale while
matching TCRMDP algorithm mechanics.

## Implementation Order

1. Add a shared TC helper module for `psi` normalization, TC parameter updates,
   and actor/critic/adversary observation construction.
2. Add TC-specific networks/losses that reuse the existing deterministic policy
   and double-Q network style.
3. Implement `vanilla_tc_m2td3` first because it validates the TC parameter
   update path and shared omniscient critic.
4. Implement `tc_rarl` second because it adds the independent adversary TD3
   replay/update path.
5. Implement `tc_m2td3` third because it shares the agent critic/replay and
   trains the adversary actor through Q.
6. Wire policy names into `learning/train.py` and config factories.

## Debug Checklist

Before training smoke tests, verify these static properties:

- `wrap_for_adv_training` is used by all new TC algorithms.
- `state.info["dr_params"]` exists after reset.
- TC step uses `next_params`, not random params, for the dynamics step.
- Replay transition stores `dynamics_params`, `next_dynamics_params`,
  `adv_action`, and separate actor/critic/adversary observations.
- Actor target and critic targets update only on delayed policy updates.
- RARL adversary loss receives negative reward.
- M2TD3 adversary update does not create a separate critic.

Verification commands should use the conda env:

```bash
conda run -n robust_rl python -m py_compile <touched files>
```

Do not use bare `pip`; use `python -m pip` inside `conda run -n robust_rl` only
if package inspection becomes necessary.

## Compatibility Notes

Recent MuJoCo Playground removed `mujoco_playground._src.collision` in favor of
contact-sensor based checks.  The custom environments now import
`custom_envs.collision.geoms_colliding`, a local MJX contact-buffer compatibility
helper, so entrypoint-level smoke tests should not depend on the removed private
module.
