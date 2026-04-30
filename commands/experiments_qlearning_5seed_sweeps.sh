#!/usr/bin/env bash
set -euo pipefail

# Five-seed q-learning sweeps for the current MuJoCo Playground configs.
#
# Default usage from distributionally_robust_learning:
#   bash commands/experiments_qlearning_5seed_sweeps.sh 0
#
# Useful filters:
#   POLICIES="td3 m2td3 tc_m2td3" bash commands/experiments_qlearning_5seed_sweeps.sh 0
#   TASK=Go1JoystickRoughTerrain POLICIES="td3 m2td3" bash commands/experiments_qlearning_5seed_sweeps.sh 0
#   USE_WANDB=true WANDB_GROUP_PREFIX=main bash commands/experiments_qlearning_5seed_sweeps.sh 0
#   ASYMMETRIC_CRITICS=true POLICIES=td3 bash commands/experiments_qlearning_5seed_sweeps.sh 0
#   DRY_RUN=true POLICIES=td3 SEEDS="1" bash commands/experiments_qlearning_5seed_sweeps.sh 0
#   SMOKE=true USE_WANDB=false POLICIES="td3 tc_rarl" bash commands/experiments_qlearning_5seed_sweeps.sh 0
#
# By default this script sweeps asymmetric_critic=true,false and keeps
# num_timesteps, num_envs, and replay sizes from learning/configs/*.py.
# SMOKE=true is the only built-in path that shrinks them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEARNING_DIR="${REPO_ROOT}/learning"

GPU_ID="${1:-${GPU_ID:-0}}"
TASK="${TASK:-CheetahRun}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ASYMMETRIC_CRITICS="${ASYMMETRIC_CRITICS:-true false}"
WANDB_PROJECT="${WANDB_PROJECT:-qlearning-5seed-sweeps}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-}"
USE_WANDB="${USE_WANDB:-true}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SAVE_AGENT="${SAVE_AGENT:-true}"
CONDA_ENV="${CONDA_ENV:-rob-q}"
USE_CONDA_RUN="${USE_CONDA_RUN:-true}"
CONDA_NO_CAPTURE_OUTPUT="${CONDA_NO_CAPTURE_OUTPUT:-true}"
PYTHON_BIN="${PYTHON_BIN:-python}"
UNSET_LD_LIBRARY_PATH="${UNSET_LD_LIBRARY_PATH:-true}"
DRY_RUN="${DRY_RUN:-false}"
SMOKE="${SMOKE:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

DEFAULT_POLICIES=(
  td3
  m2td3
  rarl
  vanilla_tc_m2td3
  tc_rarl
  tc_m2td3
)

read -r -a SELECTED_POLICIES <<< "${POLICIES:-${DEFAULT_POLICIES[*]}}"
read -r -a SELECTED_ASYMMETRIC_CRITICS <<< "${ASYMMETRIC_CRITICS}"
read -r -a EXTRA_OVERRIDE_ARGS <<< "${EXTRA_OVERRIDES}"

COMMON_OVERRIDES=(
  "task=${TASK}"
  "wandb_project=${WANDB_PROJECT}"
  "wandb_group_prefix=${WANDB_GROUP_PREFIX}"
  "use_wandb=${USE_WANDB}"
  "save_video=${SAVE_VIDEO}"
  "save_agent=${SAVE_AGENT}"
  "randomization=true"
  "eval_randomization=true"
)

SMOKE_OVERRIDES=()
if [[ "${SMOKE}" == "true" ]]; then
  SMOKE_OVERRIDES=(
    "++num_timesteps=32768"
    "++num_evals=2"
    "++num_envs=16"
    "++batch_size=64"
    "++min_replay_size=1024"
    "++max_replay_size=32768"
  )
fi

_python_cmd() {
  if [[ "${USE_CONDA_RUN}" == "true" ]]; then
    local cmd=(conda run -n "${CONDA_ENV}")
    if [[ "${CONDA_NO_CAPTURE_OUTPUT}" == "true" ]]; then
      cmd+=(--no-capture-output)
    fi
    cmd+=(python run.py)
    printf '%s\n' "${cmd[@]}"
  else
    printf '%s\n' "${PYTHON_BIN}" run.py
  fi
}

_print_cmd() {
  printf 'cd %q && ' "${LEARNING_DIR}"
  printf '%q ' "$@"
  printf '\n'
}

_wandb_group() {
  local policy="$1"
  local hp_choice="$2"
  if [[ -n "${WANDB_GROUP_PREFIX}" ]]; then
    printf '%s.%s.%s.%s\n' "${WANDB_GROUP_PREFIX}" "${TASK}" "${policy}" "${hp_choice}"
  else
    printf '%s.%s.%s\n' "${TASK}" "${policy}" "${hp_choice}"
  fi
}

run_case() {
  local policy="$1"
  local label="$2"
  local hp_choice="$3"
  shift 3
  local overrides=("$@")
  for asymmetric_critic in "${SELECTED_ASYMMETRIC_CRITICS[@]}"; do
    local asym_choice="asymmetric_critic-${asymmetric_critic}"
    local case_label="${label}_asym${asymmetric_critic}"
    local wandb_group
    wandb_group="$(_wandb_group "${policy}" "${hp_choice}_${asym_choice}")"

    for seed in ${SEEDS}; do
      mapfile -t base_cmd < <(_python_cmd)
      local env_cmd=(env)
      if [[ "${UNSET_LD_LIBRARY_PATH}" == "true" ]]; then
        env_cmd+=(-u LD_LIBRARY_PATH)
      fi
      env_cmd+=(
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false"
      )
      local cmd=(
        "${env_cmd[@]}"
        "${base_cmd[@]}"
        "policy=${policy}"
        "seed=${seed}"
        "exp_name=${case_label}"
        "comment=_${case_label}"
        "wandb_group=${wandb_group}"
        "asymmetric_critic=${asymmetric_critic}"
        "${COMMON_OVERRIDES[@]}"
        "${SMOKE_OVERRIDES[@]}"
        "${overrides[@]}"
        "${EXTRA_OVERRIDE_ARGS[@]}"
      )

      _print_cmd "${cmd[@]}"
      if [[ "${DRY_RUN}" == "true" ]]; then
        continue
      fi

      if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
        (cd "${LEARNING_DIR}" && "${cmd[@]}") || true
      else
        (cd "${LEARNING_DIR}" && "${cmd[@]}")
      fi
    done
  done
}

run_policy_sweep() {
  local policy="$1"
  case "${policy}" in
    # sac)
      # run_case sac sac_lr1e-4 "learning_rate-1e-4" "++learning_rate=1e-4"
      # run_case sac sac_lr3e-4 "learning_rate-3e-4" "++learning_rate=3e-4"
      # run_case sac sac_lr1e-3 "learning_rate-1e-3" "++learning_rate=1e-3"
      # ;;
    td3)
      run_case td3 td3_fixed_explore010 "std_min-0_1_std_max-0_1_policy_noise-0_1_noise_clip-0_5_policy_frequency-2" \
        "++std_min=0.1" "++std_max=0.1" "++policy_noise=0.1" "++noise_clip=0.5" "++policy_frequency=2"
      run_case td3 td3_range_explore040 "std_min-0_01_std_max-0_4_policy_noise-0_2_noise_clip-0_5_policy_frequency-2" \
        "++std_min=0.01" "++std_max=0.4" "++policy_noise=0.2" "++noise_clip=0.5" "++policy_frequency=2"
      run_case td3 td3_actor_every_step "std_min-0_1_std_max-0_1_policy_noise-0_2_noise_clip-0_5_policy_frequency-1" \
        "++std_min=0.1" "++std_max=0.1" "++policy_noise=0.2" "++noise_clip=0.5" "++policy_frequency=1"
      ;;
    m2td3)
      run_case m2td3 m2td3_omega005_k5 "omega_distance_threshold-0_05_num_omegas-5_omega_noise_rate-0_2_omega_clip-0_5_omega_std-1_0" \
        "++omega_distance_threshold=0.05" "++num_omegas=5"
      run_case m2td3 m2td3_omega010_k5 "omega_distance_threshold-0_1_num_omegas-5_omega_noise_rate-0_2_omega_clip-0_5_omega_std-1_0" \
        "++omega_distance_threshold=0.1" "++num_omegas=5"
      run_case m2td3 m2td3_omega010_k10 "omega_distance_threshold-0_1_num_omegas-10_omega_noise_rate-0_2_omega_clip-0_5_omega_std-1_0" \
        "++omega_distance_threshold=0.1" "++num_omegas=10"
      ;;

    rarl)
      run_case rarl rarl_omniscient_false_dr100 "omniscient_adversary-false_dr_train_ratio-1_0" \
        "omniscient_adversary=false" "dr_train_ratio=1.0"
      run_case rarl rarl_omniscient_true_dr100 "omniscient_adversary-true_dr_train_ratio-1_0" \
        "omniscient_adversary=true" "dr_train_ratio=1.0"
      run_case rarl rarl_omniscient_true_dr050 "omniscient_adversary-true_dr_train_ratio-0_5" \
        "omniscient_adversary=true" "dr_train_ratio=0.5"
      ;;

    vanilla_tc_m2td3|tc_rarl|tc_m2td3)
      run_case "${policy}" "${policy}_radius0005" "radius-0_0005" "radius=0.0005"
      run_case "${policy}" "${policy}_radius0010" "radius-0_001" "radius=0.001"
      run_case "${policy}" "${policy}_radius0020" "radius-0_002" "radius=0.002"
      ;;
    *)
      echo "Unknown policy '${policy}'. Supported: ${DEFAULT_POLICIES[*]}" >&2
      return 1
      ;;
  esac
}

for policy in "${SELECTED_POLICIES[@]}"; do
  run_policy_sweep "${policy}"
done
