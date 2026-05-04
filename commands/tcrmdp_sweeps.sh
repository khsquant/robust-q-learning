#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DR_AUGMENTED_CRITIC="${DR_AUGMENTED_CRITIC:-true}" \
POLICIES="${POLICIES:-vanilla_tc_m2td3 tc_rarl tc_m2td3}" \
exec "${SCRIPT_DIR}/whole_sweeps.sh" "$@"
