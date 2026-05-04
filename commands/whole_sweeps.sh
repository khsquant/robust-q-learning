#!/usr/bin/env bash
set -euo pipefail

# Whole q-learning sweep launcher with policy-specific sweep ranges collected
# near the top of the file. Edit the arrays in "Sweep ranges" to change what
# each algorithm tries.
#
# Usage:
#   bash commands/whole_sweeps.sh 0
#   DRY_RUN=true bash commands/whole_sweeps.sh 0
#   SMOKE=true USE_WANDB=false SEEDS="1" bash commands/whole_sweeps.sh 0
#   POLICIES="td3 m2td3" bash commands/whole_sweeps.sh 0
#   POLICIES="rarl vanilla_tc_m2td3 tc_rarl tc_m2td3" bash commands/whole_sweeps.sh 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEARNING_DIR="${REPO_ROOT}/learning"

GPU_ID="${1:-${GPU_ID:-0}}"
TASK="${TASK:-CheetahRun}"
SEEDS="${SEEDS:-1 2 3 4 5}"
DR_AUGMENTED_CRITIC="${DR_AUGMENTED_CRITIC:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-qlearning-whole-sweeps}"
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
  # m2td3
  rarl
  vanilla_tc_m2td3
  tc_rarl
  tc_m2td3
)

# Sweep ranges.
#
# Edit these arrays to change the sweep. Values are plain shell words; keep
# labels short because they are used in W&B groups and run names.

TD3_SWEEPS=(
  "default|default|"
  # "fixed_explore010|std_min-0_1_std_max-0_1_policy_noise-0_1_noise_clip-0_5_policy_frequency-2|++std_min=0.1 ++std_max=0.1 ++policy_noise=0.1 ++noise_clip=0.5 ++policy_frequency=2"
  # "range_explore040|std_min-0_01_std_max-0_4_policy_noise-0_2_noise_clip-0_5_policy_frequency-2|++std_min=0.01 ++std_max=0.4 ++policy_noise=0.2 ++noise_clip=0.5 ++policy_frequency=2"
  # "actor_every_step|std_min-0_1_std_max-0_1_policy_noise-0_2_noise_clip-0_5_policy_frequency-1|++std_min=0.1 ++std_max=0.1 ++policy_noise=0.2 ++noise_clip=0.5 ++policy_frequency=1"
)

M2TD3_SWEEPS=(
  # Keep this first: before tuning M2TD3, compare this zeroed M2-specific
  # baseline against td3/default to check whether M2TD3 reaches TD3 performance.
  # "zero_td3_check|td3_check_num_omegas-1_omega_distance_threshold-0_0_omega_noise_rate-0_0_omega_clip-0_0_omega_std-0_0_omega_lr-0_0_policy_frequency-2|++num_omegas=1 ++omega_distance_threshold=0.0 ++omega_noise_rate=0.0 ++omega_clip=0.0 ++omega_std=0.0 ++omega_lr=0.0 ++policy_frequency=2"
  "omega005_k5|omega_distance_threshold-0_05_num_omegas-5_omega_noise_rate-0_2_omega_clip-0_5_omega_std-1_0|++omega_distance_threshold=0.05 ++num_omegas=5"
  "omega010_k5|omega_distance_threshold-0_1_num_omegas-5_omega_noise_rate-0_2_omega_clip-0_5_omega_std-1_0|++omega_distance_threshold=0.1 ++num_omegas=5"
  "omega010_k10|omega_distance_threshold-0_1_num_omegas-10_omega_noise_rate-0_2_omega_clip-0_5_omega_std-1_0|++omega_distance_threshold=0.1 ++num_omegas=10"
  "omega005_k10_low_noise|omega_distance_threshold-0_05_num_omegas-10_omega_noise_rate-0_05_omega_clip-0_25_omega_std-0_25|++omega_distance_threshold=0.05 ++num_omegas=10 ++omega_noise_rate=0.05 ++omega_clip=0.25 ++omega_std=0.25"
)

RARL_SWEEPS=(
  "omniscient_false_dr100|omniscient_adversary-false_dr_train_ratio-1_0|omniscient_adversary=false dr_train_ratio=1.0"
  "omniscient_true_dr100|omniscient_adversary-true_dr_train_ratio-1_0|omniscient_adversary=true dr_train_ratio=1.0"
  "omniscient_true_dr050|omniscient_adversary-true_dr_train_ratio-0_5|omniscient_adversary=true dr_train_ratio=0.5"
)

TC_RARL_SWEEPS=(
  "radius0005|radius-0_0005|radius=0.0005"
  "radius0010|radius-0_001|radius=0.001"
  "radius0020|radius-0_002|radius=0.002"
  "radius0010_dr050|radius-0_001_dr_train_ratio-0_5|radius=0.001 dr_train_ratio=0.5"
)

VANILLA_TC_M2TD3_SWEEPS=(
  "radius0005|radius-0_0005|radius=0.0005"
  "radius0010|radius-0_001|radius=0.001"
  "radius0020|radius-0_002|radius=0.002"
  "radius0010_dr050|radius-0_001_dr_train_ratio-0_5|radius=0.001 dr_train_ratio=0.5"
)

TC_M2TD3_SWEEPS=(
  "radius0005|radius-0_0005|radius=0.0005"
  "radius0010|radius-0_001|radius=0.001"
  "radius0020|radius-0_002|radius=0.002"
  "radius0010_dr050|radius-0_001_dr_train_ratio-0_5|radius=0.001 dr_train_ratio=0.5"
)

SMOKE_OVERRIDES=()
if [[ "${SMOKE}" == "true" ]]; then
  SMOKE_OVERRIDES=(
    "++num_timesteps=4096"
    "++num_evals=1"
    "++episode_length=64"
    "++num_envs=8"
    "++num_eval_envs=16"
    "++batch_size=32"
    "++min_replay_size=256"
    "++max_replay_size=4096"
  )
fi

read -r -a SELECTED_POLICIES <<< "${POLICIES:-${DEFAULT_POLICIES[*]}}"
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
  "dr_augmented_critic=${DR_AUGMENTED_CRITIC}"
)

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

  local critic_choice="dr_augmented_critic-${DR_AUGMENTED_CRITIC}"
  local case_label="${policy}_${label}_draug${DR_AUGMENTED_CRITIC}"
  local wandb_group
  wandb_group="$(_wandb_group "${policy}" "${hp_choice}_${critic_choice}")"

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
}

run_sweep_entries() {
  local policy="$1"
  shift
  local entries=("$@")
  local entry label hp_choice override_string

  for entry in "${entries[@]}"; do
    IFS="|" read -r label hp_choice override_string <<< "${entry}"
    read -r -a overrides <<< "${override_string}"
    run_case "${policy}" "${label}" "${hp_choice}" "${overrides[@]}"
    if [[ "${SMOKE}" == "true" ]]; then
      return
    fi
  done
}

run_policy_sweep() {
  local policy="$1"
  case "${policy}" in
    td3)
      run_sweep_entries td3 "${TD3_SWEEPS[@]}"
      ;;
    # m2td3)
    #   run_sweep_entries m2td3 "${M2TD3_SWEEPS[@]}"
    #   ;;
    rarl)
      run_sweep_entries rarl "${RARL_SWEEPS[@]}"
      ;;
    vanilla_tc_m2td3)
      run_sweep_entries vanilla_tc_m2td3 "${VANILLA_TC_M2TD3_SWEEPS[@]}"
      ;;
    tc_rarl)
      run_sweep_entries tc_rarl "${TC_RARL_SWEEPS[@]}"
      ;;
    tc_m2td3)
      run_sweep_entries tc_m2td3 "${TC_M2TD3_SWEEPS[@]}"
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
