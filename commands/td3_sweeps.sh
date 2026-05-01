#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICIES="${POLICIES:-td3}" exec "${SCRIPT_DIR}/whole_sweeps.sh" "$@"
