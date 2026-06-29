#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DR_AUGMENTED_CRITIC="${DR_AUGMENTED_CRITIC:-true}" \
POLICIES="${POLICIES:-bridgetd3}" \
exec "${SCRIPT_DIR}/whole_sweeps.sh" "$@"
