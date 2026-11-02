# Monthly top-N backtest of the model signals against the S&P 500.

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modell" / "features"))
import sources  # noqa: E402
import stock  # noqa: E402

FEATURES = ROOT / "data" / "features" / "cfo_opp_features.parquet"
HOLD = 126
SEED = 42
# days on which no cohort is held (or a pick has no price) are all-NaN columns; their mean is NaN by design
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


def month_cohorts(sig: pd.DataFrame, cal: pd.DatetimeIndex, hold: int) -> pd.DataFrame:
    """Formation month -> entry and exit date; only cohorts that have closed by the last price date."""
    months = sig["signal_date"].dt.to_period("M").sort_values().unique()
    rows = []
    for m in months:
        pos = cal.searchsorted(m.end_time.normalize(), side="right")
        if pos + hold < len(cal):
            rows.append({"month": m, "entry": cal[pos], "exit": cal[pos + hold], "e": pos, "x": pos + hold})
    return pd.DataFrame(rows)


def price_paths(sig: pd.DataFrame, cohorts: pd.DataFrame, cal: pd.DatetimeIndex) -> np.ndarray:
    """Per signal: price relative to the cohort entry close (P_t / P_entry) on the calendar, NaN outside its window."""
    paths = np.full((len(sig), len(cal)), np.nan)
    co = cohorts.set_index("month")
    for sym, idx in sig.groupby("symbol").groups.items():
        adj = stock.prepare(sources.load_prices(sym), cal)["adj_close"].ffill(limit=stock.FFILL_LIMIT).to_numpy()
        for i in idx:
            m = sig.at[i, "month"]
            if m not in co.index:
                continue
            e, x = co.at[m, "e"], co.at[m, "x"]
            if np.isfinite(adj[e]) and adj[e] > 0:
                paths[sig.index.get_loc(i), e:x + 1] = adj[e:x + 1] / adj[e]
    return paths


def cohort_returns(paths: np.ndarray, members: list[np.ndarray], cohorts: pd.DataFrame, cost: float) -> np.ndarray:
    """(cohort, day) daily returns of equal-weight buy-and-hold baskets; NaN where the cohort is not held."""
    out = np.full((len(cohorts), paths.shape[1]), np.nan)
    for k, (rows, e, x) in enumerate(zip(members, cohorts["e"], cohorts["x"])):
        if len(rows) == 0:
            continue
        value = np.nanmean(paths[rows, e:x + 1], axis=0) * (1 - cost)
        value[-1] *= 1 - cost
        out[k, e:x + 1] = np.diff(np.r_[1.0, value]) / np.r_[1.0, value[:-1]]
    return out


def portfolio(cohort_ret: np.ndarray) -> np.ndarray:
    return np.nanmean(cohort_ret, axis=0)


def metrics(r: pd.Series, rf: pd.Series, mkt: pd.Series | None = None) -> dict:
    ex = r - rf
    down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    wealth = (1 + r).cumprod()
    years = len(r) / 252
    out = {
        "total_return": float(wealth.iloc[-1] - 1),
        "cagr": float(wealth.iloc[-1] ** (1 / years) - 1),
        "ann_vol": float(r.std() * np.sqrt(252)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)),
        "sortino": float(ex.mean() / down * np.sqrt(252)),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
    }
    if mkt is not None:
        mx = mkt - rf
        beta = float(np.cov(ex, mx)[0, 1] / mx.var())
        out["beta"] = beta
        out["alpha_ann"] = float((ex.mean() - beta * mx.mean()) * 252)
        active = r - mkt
        out["information_ratio"] = float(active.mean() / active.std() * np.sqrt(252))
    return out


def cohort_table(paths, members, cohorts, spy_adj) -> pd.DataFrame:
    rows = []
    for (m, e, x), rr in zip(cohorts[["month", "e", "x"]].itertuples(index=False), members):
        ret = float(np.nanmean(paths[rr, x]) - 1) if len(rr) else np.nan
        spy = float(spy_adj[x] / spy_adj[e] - 1)
        rows.append({"month": str(m), "n": len(rr), "ret": ret, "spy": spy, "excess": ret - spy})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "data" / "model" / "xgb_train2016")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--hold", type=int, default=HOLD)
    ap.add_argument("--cost-bps", type=float, default=0.0, help="trading cost per side in basis points")
    ap.add_argument("--draws", type=int, default=1000, help="random top-N portfolios for the chance distribution")
    args = ap.parse_args(argv)
    cost = args.cost_bps / 1e4

    cal = stock.market_calendar()
    sig = pd.read_parquet(args.model_dir / "test_predictions.parquet")
    sig = sig.merge(pd.read_parquet(FEATURES, columns=["event_id", "stk_vol_3m"]), on="event_id")
    sig["month"] = sig["signal_date"].dt.to_period("M")
    cohorts = month_cohorts(sig, cal, args.hold)
    sig = sig[sig["month"].isin(cohorts["month"])].reset_index(drop=True)
    paths = price_paths(sig, cohorts, cal)
    has_price = np.isfinite(paths).any(axis=1)

    def pick(score: pd.Series, n: int | None) -> list[np.ndarray]:
        """Per cohort: the rows of the n highest-scoring distinct issuers with an entry price (all if n is None)."""
        out = []
        for m in cohorts["month"]:
            s = sig[(sig["month"] == m) & has_price].assign(score=score).sort_values("score", ascending=False)
            s = s.drop_duplicates("issuer_cik")
            out.append(s.index.to_numpy()[:n] if n else s.index.to_numpy())
        return out

    first, last = cohorts["e"].iloc[0], cohorts["x"].iloc[-1]
    days = cal[first:last + 1]
    rf = sources.fred_daily(["DTB3"])["DTB3"].dropna().reindex(days, method="ffill") / 100 / 252
    spy_adj = stock.prepare(sources.load_prices(stock.MARKET_ETF), cal)["adj_close"].ffill().to_numpy()
    spy = pd.Series(spy_adj[first:last + 1], index=days).pct_change().fillna(0)

    strategies = {
        f"model top {args.top}": pick(sig["pred"], args.top),
        f"lowest volatility {args.top}": pick(-sig["stk_vol_3m"].fillna(np.inf), args.top),
        "all opportunistic buys": pick(sig["pred"], None),
    }
    results, tables = {}, {}
    for name, members in strategies.items():
        r = pd.Series(portfolio(cohort_returns(paths, members, cohorts, cost))[first:last + 1], index=days).fillna(0)
        results[name] = metrics(r, rf, spy)
        tables[name] = cohort_table(paths, members, cohorts, spy_adj)
        results[name]["cohort_mean_excess_6m"] = float(tables[name]["excess"].mean())
        results[name]["cohort_hit_rate_vs_spy"] = float((tables[name]["excess"] > 0).mean())
    results["SPY buy and hold"] = metrics(spy, rf)

    rng = np.random.default_rng(SEED)
    rand = []
    for _ in range(args.draws):
        members = pick(pd.Series(rng.random(len(sig))), args.top)
        r = pd.Series(portfolio(cohort_returns(paths, members, cohorts, cost))[first:last + 1], index=days).fillna(0)
        rand.append(metrics(r, rf, spy))
    rand = pd.DataFrame(rand)
    model = results[f"model top {args.top}"]
    chance = {k: {"median": float(rand[k].median()), "p05": float(rand[k].quantile(0.05)),
                  "p95": float(rand[k].quantile(0.95)), "model_percentile": float((rand[k] < model[k]).mean())}
              for k in ("cagr", "sharpe", "sortino", "alpha_ann", "max_drawdown")}

    report = {
        "model_dir": str(args.model_dir), "top": args.top, "hold_days": args.hold, "cost_bps_per_side": args.cost_bps,
        "period": [str(days[0].date()), str(days[-1].date())],
        "formation_months": [str(cohorts["month"].iloc[0]), str(cohorts["month"].iloc[-1])],
        "n_cohorts": len(cohorts), "signals_per_month_median": float(sig.groupby("month").size().median()),
        "strategies": results, "random_top_n": chance,
    }
    out_dir = args.model_dir / "backtest"
    out_dir.mkdir(exist_ok=True)
    tag = f"top{args.top}_cost{args.cost_bps:g}"
    (out_dir / f"backtest_{tag}.json").write_text(json.dumps(report, indent=2))
    pd.concat(tables, names=["strategy"]).to_csv(out_dir / f"cohorts_{tag}.csv")

    pd.set_option("display.width", 200)
    print(f"period {report['period'][0]} .. {report['period'][1]}, {len(cohorts)} monthly cohorts "
          f"({report['formation_months'][0]} .. {report['formation_months'][1]}), cost {args.cost_bps:g} bps/side")
    print(pd.DataFrame(results).T.round(4).to_string())
    print(f"\nrandom {args.top} per month, {args.draws} draws (model percentile = share of random portfolios below "
          f"the model):")
    print(pd.DataFrame(chance).T.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
