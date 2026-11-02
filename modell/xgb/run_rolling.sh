#!/bin/sh
# Rolling yearly retraining of config B (modell/xgb/rolling.py).
set -e
cd "$(dirname "$0")/../.."
export XGB_N_JOBS=2
mkdir -p logs
for s in 42 1 2 3 4; do
  .venv/bin/python modell/xgb/rolling.py --seeds $s --fit-only > logs/rolling_seed$s.log 2>&1 &
done
wait
.venv/bin/python modell/xgb/rolling.py --seeds 42 1 2 3 4 > logs/rolling.log 2>&1
