#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICIES="${POLICIES:-tc_rarl}" exec "${SCRIPT_DIR}/experiments_qlearning_5seed_sweeps.sh" "$@"
