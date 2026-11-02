#!/bin/sh
# Multi-seed robustness: reruns of config B and C with other seeds, then the backtests.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=data/model/robust
for s in ${SEEDS:-1 2 3 4 5 6 7 8 9}; do
  for cfg in B C; do
    d=$OUT/${cfg}_seed$s
    if [ ! -f $d/report.json ]; then
      if [ $cfg = B ]; then
        $PY modell/xgb/train.py --trials 100 --train-start 2016-01-01 --val-years 2018-2021 --seed $s --out-dir $d
      else
        $PY modell/xgb/selection.py --trials 100 --seed $s --out-dir $d
      fi
    fi
    for top in 5 10 20; do
      [ -f $d/backtest/backtest_top${top}_cost0.json ] || $PY modell/xgb/backtest.py --model-dir $d --top $top --draws 200
    done
    [ -f $d/backtest/backtest_top5_cost25.json ] || $PY modell/xgb/backtest.py --model-dir $d --top 5 --cost-bps 25 --draws 200
  done
done
