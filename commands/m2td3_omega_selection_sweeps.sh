#!/usr/bin/env bash
set -euo pipefail

# M2TD3 omega-selection sweep launcher.
#
# This script tunes the TCRMDP-style omega selection and restart knobs:
#   ++omega_min_probability
#   ++omega_prob_update_rate
#   ++omega_restart_distance
#   ++omega_restart_probability
#
# Usage:
#   bash commands/m2td3_omega_selection_sweeps.sh 0
#   DRY_RUN=true bash commands/m2td3_omega_selection_sweeps.sh 0
#   SWEEP_MODE=grid DRY_RUN=true bash commands/m2td3_omega_selection_sweeps.sh 0
#   SMOKE=true USE_WANDB=false SEEDS="1" ASYMMETRIC_CRITICS=true bash commands/m2td3_omega_selection_sweeps.sh 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEARNING_DIR="${REPO_ROOT}/learning"

GPU_ID="${1:-${GPU_ID:-0}}"
TASK="${TASK:-CheetahRun}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ASYMMETRIC_CRITICS="${ASYMMETRIC_CRITICS:-true false}"
WANDB_PROJECT="${WANDB_PROJECT:-qlearning-m2td3-omega-selection-sweeps}"
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

# one_factor: baseline plus one-at-a-time sweeps.
# grid: full Cartesian product of the ranges below.
SWEEP_MODE="${SWEEP_MODE:-one_factor}"

# Base M2TD3 setting to hold fixed while tuning omega selection.
# Edit this first if you want to tune around a different M2TD3 baseline.
BASE_M2TD3_OVERRIDES=(
  "++num_omegas=5"
  "++omega_distance_threshold=0.1"
  "++omega_noise_rate=0.2"
  "++omega_clip=0.5"
  "++omega_std=1.0"
  "++policy_frequency=2"
)

# Default point for one-factor sweeps.
DEFAULT_OMEGA_MIN_PROBABILITY="${DEFAULT_OMEGA_MIN_PROBABILITY:-0.05}"
DEFAULT_OMEGA_PROB_UPDATE_RATE="${DEFAULT_OMEGA_PROB_UPDATE_RATE:-default}"
DEFAULT_OMEGA_RESTART_DISTANCE="${DEFAULT_OMEGA_RESTART_DISTANCE:-true}"
DEFAULT_OMEGA_RESTART_PROBABILITY="${DEFAULT_OMEGA_RESTART_PROBABILITY:-true}"

# Sweep ranges. Keep labels short because they are used in W&B groups.
OMEGA_MIN_PROBABILITIES=(
  0.0
  0.01
  0.05
  0.10
)

# Use "default" to let run.py set 1 / episode_length, matching the original
# TCRMDP update coefficient idea.
OMEGA_PROB_UPDATE_RATES=(
  default
  0.0005
  0.001
  0.005
  0.01
)

# Format: label|omega_restart_distance|omega_restart_probability
OMEGA_RESTART_MODES=(
  "both|true|true"
  "distance_only|true|false"
  "probability_only|false|true"
  "none|false|false"
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

_label_value() {
  local value="$1"
  value="${value//./_}"
  value="${value//-/_neg_}"
  printf '%s' "${value}"
}

_wandb_group() {
  local hp_choice="$1"
  if [[ -n "${WANDB_GROUP_PREFIX}" ]]; then
    printf '%s.%s.m2td3.%s\n' "${WANDB_GROUP_PREFIX}" "${TASK}" "${hp_choice}"
  else
    printf '%s.m2td3.%s\n' "${TASK}" "${hp_choice}"
  fi
}

_run_case_count=0

run_case() {
  local label="$1"
  local hp_choice="$2"
  shift 2
  local overrides=("$@")

  for asymmetric_critic in "${SELECTED_ASYMMETRIC_CRITICS[@]}"; do
    local asym_choice="asymmetric_critic-${asymmetric_critic}"
    local case_label="m2td3_${label}_asym${asymmetric_critic}"
    local wandb_group
    wandb_group="$(_wandb_group "${hp_choice}_${asym_choice}")"

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
        "policy=m2td3"
        "seed=${seed}"
        "exp_name=${case_label}"
        "comment=_${case_label}"
        "wandb_group=${wandb_group}"
        "asymmetric_critic=${asymmetric_critic}"
        "${COMMON_OVERRIDES[@]}"
        "${SMOKE_OVERRIDES[@]}"
        "${BASE_M2TD3_OVERRIDES[@]}"
        "${overrides[@]}"
        "${EXTRA_OVERRIDE_ARGS[@]}"
      )

      _print_cmd "${cmd[@]}"
      _run_case_count=$((_run_case_count + 1))
      if [[ "${DRY_RUN}" == "true" ]]; then
        if [[ "${SMOKE}" == "true" ]]; then
          exit 0
        fi
        continue
      fi

      if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
        (cd "${LEARNING_DIR}" && "${cmd[@]}") || true
      else
        (cd "${LEARNING_DIR}" && "${cmd[@]}")
      fi

      if [[ "${SMOKE}" == "true" ]]; then
        exit 0
      fi
    done
  done
}

run_omega_case() {
  local label="$1"
  local min_probability="$2"
  local prob_update_rate="$3"
  local restart_label="$4"
  local restart_distance="$5"
  local restart_probability="$6"

  local min_label update_label hp_choice
  min_label="$(_label_value "${min_probability}")"
  update_label="$(_label_value "${prob_update_rate}")"
  hp_choice="ominp-${min_label}_oupd-${update_label}_orst-${restart_label}"

  local overrides=(
    "++omega_min_probability=${min_probability}"
    "++omega_restart_distance=${restart_distance}"
    "++omega_restart_probability=${restart_probability}"
  )
  if [[ "${prob_update_rate}" != "default" ]]; then
    overrides+=("++omega_prob_update_rate=${prob_update_rate}")
  fi

  run_case "${label}" "${hp_choice}" "${overrides[@]}"
}

run_one_factor_sweep() {
  run_omega_case \
    "omega_sel_default" \
    "${DEFAULT_OMEGA_MIN_PROBABILITY}" \
    "${DEFAULT_OMEGA_PROB_UPDATE_RATE}" \
    "both" \
    "${DEFAULT_OMEGA_RESTART_DISTANCE}" \
    "${DEFAULT_OMEGA_RESTART_PROBABILITY}"

  local min_probability
  for min_probability in "${OMEGA_MIN_PROBABILITIES[@]}"; do
    [[ "${min_probability}" == "${DEFAULT_OMEGA_MIN_PROBABILITY}" ]] && continue
    run_omega_case \
      "ominp_$(_label_value "${min_probability}")" \
      "${min_probability}" \
      "${DEFAULT_OMEGA_PROB_UPDATE_RATE}" \
      "both" \
      "${DEFAULT_OMEGA_RESTART_DISTANCE}" \
      "${DEFAULT_OMEGA_RESTART_PROBABILITY}"
  done

  local prob_update_rate
  for prob_update_rate in "${OMEGA_PROB_UPDATE_RATES[@]}"; do
    [[ "${prob_update_rate}" == "${DEFAULT_OMEGA_PROB_UPDATE_RATE}" ]] && continue
    run_omega_case \
      "oupd_$(_label_value "${prob_update_rate}")" \
      "${DEFAULT_OMEGA_MIN_PROBABILITY}" \
      "${prob_update_rate}" \
      "both" \
      "${DEFAULT_OMEGA_RESTART_DISTANCE}" \
      "${DEFAULT_OMEGA_RESTART_PROBABILITY}"
  done

  local mode restart_label restart_distance restart_probability
  for mode in "${OMEGA_RESTART_MODES[@]}"; do
    IFS="|" read -r restart_label restart_distance restart_probability <<< "${mode}"
    [[ "${restart_distance}" == "${DEFAULT_OMEGA_RESTART_DISTANCE}" \
      && "${restart_probability}" == "${DEFAULT_OMEGA_RESTART_PROBABILITY}" ]] && continue
    run_omega_case \
      "orst_${restart_label}" \
      "${DEFAULT_OMEGA_MIN_PROBABILITY}" \
      "${DEFAULT_OMEGA_PROB_UPDATE_RATE}" \
      "${restart_label}" \
      "${restart_distance}" \
      "${restart_probability}"
  done
}

run_grid_sweep() {
  local min_probability prob_update_rate mode restart_label restart_distance restart_probability
  for min_probability in "${OMEGA_MIN_PROBABILITIES[@]}"; do
    for prob_update_rate in "${OMEGA_PROB_UPDATE_RATES[@]}"; do
      for mode in "${OMEGA_RESTART_MODES[@]}"; do
        IFS="|" read -r restart_label restart_distance restart_probability <<< "${mode}"
        run_omega_case \
          "ominp_$(_label_value "${min_probability}")_oupd_$(_label_value "${prob_update_rate}")_orst_${restart_label}" \
          "${min_probability}" \
          "${prob_update_rate}" \
          "${restart_label}" \
          "${restart_distance}" \
          "${restart_probability}"
      done
    done
  done
}

case "${SWEEP_MODE}" in
  one_factor)
    run_one_factor_sweep
    ;;
  grid)
    run_grid_sweep
    ;;
  *)
    echo "Unknown SWEEP_MODE='${SWEEP_MODE}'. Supported: one_factor, grid" >&2
    exit 1
    ;;
esac
