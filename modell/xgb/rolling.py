# Rolling yearly retraining of config B.

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import robustness as rb  # noqa: E402
import train  # noqa: E402

OUT = ROOT / "data" / "model" / "rolling"
TRAIN_YEARS = 6
VAL_YEARS = 4
PERIODS = {"all": ("2012-01", "2026-12"), "2012-2021": ("2012-01", "2021-12"), "2022-2026": ("2022-01", "2026-12")}

log = logging.getLogger("xgb.rolling")


def run_year(seed: int, year: int, trials: int) -> Path:
    d = OUT / f"seed{seed}" / str(year)
    if not (d / "report.json").exists():
        t0 = time.time()
        train.main(["--trials", str(trials), "--seed", str(seed), "--out-dir", str(d),
                    "--train-start", f"{year - TRAIN_YEARS}-01-01", "--test-start", f"{year}-01-01",
                    "--val-years", f"{year - VAL_YEARS}-{year - 1}"])
        log.info("seed %d, test year %d: %.0fs", seed, year, time.time() - t0)
    return d


def join_years(seed: int, years: list[int]) -> Path:
    parts = []
    for y in years:
        p = pd.read_parquet(OUT / f"seed{seed}" / str(y) / "test_predictions.parquet")
        parts.append(p[p["signal_date"].dt.year == y])
    d = OUT / f"seed{seed}"
    pd.concat(parts, ignore_index=True).to_parquet(d / "test_predictions.parquet", index=False)
    return d


def period_tests(d: Path, top: int, cost: float) -> list[dict]:
    """Model minus SPY / all CFO buys / lowest volatility per period, paired by cohort (Newey-West)."""
    co = pd.read_csv(d / "backtest" / f"cohorts_top{top}_cost{cost:g}.csv").pivot(
        index="month", columns="strategy", values="excess")
    rows = []
    for name, (a, b) in PERIODS.items():
        c = co[(co.index >= a) & (co.index <= b)]
        model = c[f"model top {top}"].to_numpy()
        row = {"period": name, "n_cohorts": len(c), "model_excess_6m": float(np.nanmean(model))}
        for base, col in (("spy", None), ("all", "all opportunistic buys"), ("lowvol", f"lowest volatility {top}")):
            nw = rb.newey_west(model - (c[col].to_numpy() if col else 0.0))
            row[f"vs_{base}_mean"], row[f"vs_{base}_t"] = nw["mean"], nw["t"]
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--first-year", type=int, default=2012)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--fit-only", action="store_true", help="only train the yearly models")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    years = list(range(args.first_year, args.last_year + 1))

    dirs = []
    for seed in args.seeds:
        for y in years:
            run_year(seed, y, args.trials)
        if not args.fit_only:
            dirs.append((seed, join_years(seed, years)))
    if args.fit_only:
        return 0
    ens = OUT / "ensemble"
    ens.mkdir(parents=True, exist_ok=True)
    preds = [pd.read_parquet(d / "test_predictions.parquet") for _, d in dirs]
    base = preds[0]
    for (_, d), p in zip(dirs, preds):
        if not p["event_id"].equals(base["event_id"]):
            raise ValueError(f"{d}: test signals differ")
    month = base["signal_date"].dt.to_period("M")
    ranks = pd.concat([p["pred"].groupby(month).rank(pct=True) for p in preds], axis=1).mean(axis=1)
    base.assign(pred=ranks).to_parquet(ens / "test_predictions.parquet", index=False)
    (ens / "report.json").write_text(json.dumps({"seeds": args.seeds, "years": years}, indent=2))

    rows, periods = [], []
    for seed, d in dirs + [("ensemble", ens)]:
        for top, cost in rb.PORTFOLIOS:
            bt.main(["--model-dir", str(d), "--top", str(top), "--cost-bps", str(cost), "--draws", str(args.draws)])
            rows.append({"seed": seed, **rb.evaluate_run(d, top, cost)})
            if cost == 0:
                periods += [{"seed": seed, "top": top, **r} for r in period_tests(d, top, cost)]
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    per = pd.DataFrame(periods)
    per.to_csv(OUT / "summary.csv", index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print()
    print(per.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
