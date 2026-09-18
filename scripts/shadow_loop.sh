#!/bin/bash
# Keep one GPU busy building shadows until the stack is complete.
#
#   shadow_loop.sh <gpu> <config> <attack-label> <K> [direction]
#
# A plain worker is handed a fixed range and exits when that range is claimed,
# which leaves its GPU idle while another worker is still grinding through its
# own share.  This wrapper re-runs the worker over the whole range until every
# shadow's features exist, so a GPU that finishes early picks up the remainder.
#
# `direction` is "up" (default) or "down"; running one loop per GPU in opposite
# directions keeps them working on different shadows for as long as possible.
# Collisions are impossible either way -- workers claim shadows with lock files.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=$1; CONFIG=$2; LABEL=$3; K=$4; DIRECTION=${5:-up}
PY=/home/golobs/miniconda3/envs/recon_/bin/python

if [ "$DIRECTION" = "down" ]; then
    RANGE=$(seq "$K" -1 1)
else
    RANGE=$(seq 1 "$K")
fi

# The features directory to count: resolved from the config so the loop does not
# have to know how cache paths are built.
CACHE=$($PY - "$CONFIG" "$LABEL" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from mia.experiment import Experiment
exp = Experiment.load(sys.argv[1])
print(exp.build_attack(sys.argv[2]).cache(exp.dataset) / "features")
PYEOF
)

while true; do
    have=$(ls "$CACHE"/*.npz 2>/dev/null | wc -l)
    if [ "$have" -ge "$K" ]; then
        echo "### all $K shadows present at $(date +%H:%M:%S)"
        break
    fi
    echo "### [$(date +%H:%M:%S)] $have/$K shadows; GPU $GPU taking another pass"
    ./scripts/shadow_worker.sh "$GPU" "$CONFIG" "$LABEL" $RANGE
    sleep 30
done
