# Two extra tests on the rolling out-of-sample predictions of config B (data/model/rolling/, 2012-2026).

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "modell" / "features"))
import backtest as bt  # noqa: E402
import rolling  # noqa: E402
import sources  # noqa: E402
import stock  # noqa: E402
from robustness import newey_west  # noqa: E402

OUT = rolling.OUT
RUNS = ["seed42", "seed1", "seed2", "seed3", "seed4", "ensemble"]
BENCHMARKS = ["SPY", "IWM", "IJR", "RSP"]
MIN_SIGNALS = 10


def by_period(x: pd.Series) -> dict:
    out = {}
    for name, (a, b) in rolling.PERIODS.items():
        v = x[(x.index >= a) & (x.index <= b)].to_numpy(float)
        nw = newey_west(v)
        out[name] = (nw["mean"], nw["t"], nw["n_cohorts"])
    return out


def cohort_signal_returns(run: Path, cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per signal: its cohort month and the 6-month return from the cohort entry to the exit."""
    sig = pd.read_parquet(run / "test_predictions.parquet")
    sig = sig.merge(pd.read_parquet(bt.FEATURES, columns=["event_id", "stk_vol_3m"]), on="event_id")
    sig["month"] = sig["signal_date"].dt.to_period("M")
    cohorts = bt.month_cohorts(sig, cal, bt.HOLD)
    sig = sig[sig["month"].isin(cohorts["month"])].reset_index(drop=True)
    paths = bt.price_paths(sig, cohorts, cal)
    x = sig["month"].map(cohorts.set_index("month")["x"]).to_numpy()
    sig["ret"] = paths[np.arange(len(sig)), x] - 1
    return sig, cohorts


def long_short(sig: pd.DataFrame, score: str, stat: str = "mean") -> pd.Series:
    rows = {}
    for m, g in sig.groupby("month"):
        g = g.sort_values("pred", ascending=False).drop_duplicates("issuer_cik")
        g = g[np.isfinite(g["ret"]) & g[score].notna()]
        if len(g) < MIN_SIGNALS:
            continue
        q = np.ceil(g[score].rank(pct=True, method="first") * 5)
        rows[str(m)] = g.loc[q == 5, "ret"].agg(stat) - g.loc[q == 1, "ret"].agg(stat)
    return pd.Series(rows, dtype=float)


def main() -> int:
    cal = stock.market_calendar()
    bench_adj = {b: stock.prepare(sources.load_prices(b), cal)["adj_close"].ffill().to_numpy() for b in BENCHMARKS}
    bench_rows, ls_rows, ls_cohorts = [], [], {}
    for name in RUNS:
        run = OUT / name
        sig, cohorts = cohort_signal_returns(run, cal)
        co = cohorts.assign(month=cohorts["month"].astype(str)).set_index("month")
        bench = pd.DataFrame({b: adj[co["x"]] / adj[co["e"]] - 1 for b, adj in bench_adj.items()}, index=co.index)

        # 1. benchmarks: the long-only portfolios of backtest.py against each index
        for top in (5, 10, 20):
            c = pd.read_csv(run / "backtest" / f"cohorts_top{top}_cost0.csv").pivot(
                index="month", columns="strategy", values="ret")
            for strat, label in ((f"model top {top}", f"model top {top}"), ("all opportunistic buys", "all CFO buys")):
                if top != 5 and label == "all CFO buys":
                    continue
                for b in BENCHMARKS:
                    d = (c[strat] - bench[b]).dropna()
                    for period, (mean, t, n) in by_period(d).items():
                        bench_rows.append({"run": name, "portfolio": label, "benchmark": b, "period": period,
                                           "mean_excess_6m": mean, "t": t, "n_cohorts": n})

        # 2. long-short quintiles inside the CFO buys
        sig["neg_vol"] = -sig["stk_vol_3m"]
        for variant in ("mean", "median"):
            model, lowvol = long_short(sig, "pred", variant), long_short(sig, "neg_vol", variant)
            ls_cohorts[(name, variant)] = pd.DataFrame({"model": model, "lowvol": lowvol})
            for label, x in (("model Q5-Q1", model), ("low volatility Q5-Q1", lowvol),
                             ("model minus low volatility", (model - lowvol).dropna())):
                for period, (mean, t, n) in by_period(x).items():
                    ls_rows.append({"run": name, "variant": variant, "spread": label, "period": period,
                                    "mean_6m": mean, "t": t, "n_cohorts": n})
        print(f"{name}: done")

    bench_df, ls_df = pd.DataFrame(bench_rows), pd.DataFrame(ls_rows)
    bench_df.to_csv(OUT / "benchmarks.csv", index=False)
    ls_df.to_csv(OUT / "long_short.csv", index=False)
    pd.concat(ls_cohorts, names=["run", "variant", "month"]).to_csv(OUT / "long_short_cohorts.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    seeds = [r for r in RUNS if r != "ensemble"]
    for df, keys, val in ((bench_df, ["portfolio", "benchmark", "period"], "mean_excess_6m"),
                          (ls_df, ["variant", "spread", "period"], "mean_6m")):
        s = df[df["run"].isin(seeds)].groupby(keys).agg(mean=(val, "median"), t_median=("t", "median"),
                                                        t_min=("t", "min"), t_max=("t", "max"))
        e = df[df["run"] == "ensemble"].set_index(keys)[[val, "t"]].rename(
            columns={val: "ens_mean", "t": "ens_t"})
        print(s.join(e).round(3).to_string(), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
