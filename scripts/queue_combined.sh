#!/bin/bash
# Start the COMBINED grid once the BRCA orchestrator has released both GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "orchestrate_brca" > /dev/null; do sleep 120; done
exec ./scripts/orchestrate_combined.sh
