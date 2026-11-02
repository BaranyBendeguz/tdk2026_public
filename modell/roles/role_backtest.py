# Model-free backtest of insider purchases by role group.

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for d in ("modell/features", "modell/xgb", "data/opportunistic_buys"):
    sys.path.insert(0, str(ROOT / d))
import backtest as bt  # noqa: E402
import build_features as bf  # noqa: E402
import sources  # noqa: E402
import stock  # noqa: E402
from opp_buy import add_buy_patterns  # noqa: E402
from opp_filter import load_purchases  # noqa: E402

OUT = ROOT / "data" / "roles"
GROUPS = {1: "ceo", 2: "cfo", 3: "other_officer", 4: "other_officer", 5: "other_officer", 6: "director",
          7: "ten_pct_owner"}
ORDER = ["ceo", "cfo", "other_officer", "director", "ten_pct_owner"]
PERIODS = {"full": "2006-01", "test": "2022-01"}
SIGNAL_KEY = ["owner_cik", "issuer_cik", "signal_date", "is_opportunistic"]
HOLD = 126
MIN_PRICE = 1.0
SEED = 42
NW_LAGS = HOLD // 21 - 1
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

log = logging.getLogger("roles")


def build_signals() -> pd.DataFrame:
    rows = add_buy_patterns(load_purchases())
    rows = rows[rows["prev_episodes"].notna() & rows["role_rank"].isin(GROUPS)]
    ev = bf.build_events(rows[bf.ROW_COLS].copy())
    lab = (rows.assign(trans_date=pd.to_datetime(rows["trans_date"]))
           .groupby(bf.EVENT_KEY).agg(role_rank=("role_rank", "min"), is_opportunistic=("is_opportunistic", "max")))
    ev = ev.merge(lab.reset_index(), on=bf.EVENT_KEY)
    ev["group"] = ev["role_rank"].map(GROUPS)
    log.info("%d purchase rows -> %d events (%d opportunistic)", len(rows), len(ev), int(ev["is_opportunistic"].sum()))

    cands = stock.candidate_symbols(ev, sources.sec_company_tickers())
    sources.yahoo_prices(list(cands["symbol"].unique()) + [stock.MARKET_ETF])
    cal = stock.market_calendar()
    ev["anchor"] = stock.anchor_dates(ev["trans_date"], cal)
    ev = ev.merge(stock.resolve_symbols(ev, cands, cal)[["event_id", "symbol", "symbol_status"]], on="event_id")

    sig = ev[ev["symbol"].notna() & (ev["only_deriv"] == 0)]
    sig = sig.sort_values(SIGNAL_KEY + ["trans_date"]).drop_duplicates(SIGNAL_KEY, keep="last")
    cols = ["event_id", "owner_cik", "issuer_cik", "trans_date", "signal_date", "anchor", "group", "role_rank",
            "is_opportunistic", "symbol", "symbol_status", "value_usd"]
    return sig[cols].reset_index(drop=True)


def cohort_frame(cal: pd.DatetimeIndex, first: str, last_month: pd.Period) -> pd.DataFrame:
    months = pd.period_range(first, last_month, freq="M")
    e = cal.searchsorted(months.end_time.normalize(), side="right")
    ok = e + HOLD < len(cal)
    return pd.DataFrame({"month": months[ok], "e": e[ok], "x": e[ok] + HOLD})


def price_segments(sig: pd.DataFrame, cal: pd.DatetimeIndex, max_move: float) -> tuple[np.ndarray, int]:
    """Per signal: P_t / P_entry over its cohort's HOLD + 1 days (NaN if no entry price, price < MIN_PRICE, or a
    one-day move above max_move times; 0 = no limit). Also returns the number of signals dropped for the move."""
    seg = np.full((len(sig), HOLD + 1), np.nan, dtype=np.float32)
    e_all = cal.searchsorted(sig["signal_date"].dt.to_period("M").dt.end_time.dt.normalize().to_numpy(),
                             side="right")
    a_all = cal.searchsorted(sig["anchor"].to_numpy(), side="left")
    for n, (sym, idx) in enumerate(sig.groupby("symbol").groups.items()):
        p = stock.prepare(sources.load_prices(sym), cal)
        adj = p["adj_close"].ffill(limit=stock.FFILL_LIMIT).to_numpy()
        raw = p["raw_close"].ffill(limit=stock.FFILL_LIMIT).to_numpy()
        for i in sig.index.get_indexer(idx):
            e, a = e_all[i], a_all[i]
            if e + HOLD < len(cal) and a < len(cal) and raw[a] >= MIN_PRICE and np.isfinite(adj[e]) and adj[e] > 0:
                seg[i] = adj[e:e + HOLD + 1] / adj[e]
        if n % 2000 == 0:
            log.info("price segments: %d / %d symbols", n, sig["symbol"].nunique())
    dropped = 0
    if max_move > 0:
        with np.errstate(all="ignore"):
            move = np.nanmax(np.abs(np.log(seg[:, 1:] / seg[:, :-1])), axis=1, initial=0.0)
        bad = move > np.log(max_move)
        seg[bad] = np.nan
        dropped = int(bad.sum())
    return seg, dropped


def basket_returns(seg: np.ndarray, members: list[np.ndarray], cohorts: pd.DataFrame, n_days: int, start: int,
                   cost: float) -> np.ndarray:
    """Daily portfolio returns over the period days (start = calendar index of the first day)."""
    out = np.full((len(cohorts), n_days), np.nan)
    for k, (rows, e) in enumerate(zip(members, cohorts["e"])):
        if len(rows) == 0:
            continue
        value = np.nanmean(seg[rows], axis=0).astype(float) * (1 - cost)
        value[-1] *= 1 - cost
        out[k, e - start:e - start + HOLD + 1] = np.diff(np.r_[1.0, value]) / np.r_[1.0, value[:-1]]
    return np.nan_to_num(np.nanmean(out, axis=0))


def newey_west(x: np.ndarray, lags: int = NW_LAGS) -> dict:
    x = x[np.isfinite(x)]
    n, m = len(x), x.mean()
    d = x - m
    var = d @ d / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    se = np.sqrt(var / n)
    return {"mean": float(m), "se": float(se), "t": float(m / se), "n_cohorts": int(n)}


def run_period(sig: pd.DataFrame, seg: np.ndarray, cal: pd.DatetimeIndex, first: str, top: int, draws: int,
               cost: float, rf_all: pd.Series, spy_adj: np.ndarray) -> tuple[dict, pd.DataFrame]:
    has = np.isfinite(seg[:, -1]) | np.isfinite(seg).any(axis=1)
    last_month = sig["signal_date"].max().to_period("M")
    cohorts = cohort_frame(cal, first, last_month)
    start, end = cohorts["e"].iloc[0], cohorts["x"].iloc[-1]
    days = cal[start:end + 1]
    rf = rf_all.reindex(days, method="ffill")
    spy = pd.Series(spy_adj[start:end + 1], index=days).pct_change().fillna(0)
    spy_6m = spy_adj[cohorts["x"].to_numpy()] / spy_adj[cohorts["e"].to_numpy()] - 1
    month = sig["signal_date"].dt.to_period("M").to_numpy()
    rng = np.random.default_rng(SEED)

    res, tables, excess = {"SPY buy and hold": bt.metrics(spy, rf)}, [], {}
    for g in ORDER:
        res[g] = {}
        cand = {}
        for label, opp in (("opportunistic", True), ("routine", False)):
            m = (sig["group"] == g).to_numpy() & (sig["is_opportunistic"] == opp).to_numpy() & has
            members = []
            for cm in cohorts["month"]:
                r = np.flatnonzero(m & (month == cm))
                _, first_of_issuer = np.unique(sig["issuer_cik"].to_numpy()[r], return_index=True)
                members.append(r[np.sort(first_of_issuer)])
            if label == "opportunistic":
                cand = members
            r = pd.Series(basket_returns(seg, members, cohorts, len(days), start, cost), index=days)
            met = bt.metrics(r, rf, spy)
            ret6 = np.array([np.nanmean(seg[rr, -1]) - 1 if len(rr) else np.nan for rr in members])
            exc = ret6 - spy_6m
            excess[(g, label)] = exc
            met["n_signals"] = int(sum(len(rr) for rr in members))
            met["issuers_per_cohort_median"] = float(np.median([len(rr) for rr in members]))
            met["cohort_excess_6m"] = newey_west(exc)
            met["cohort_hit_rate_vs_spy"] = float(np.nanmean(exc > 0))
            res[g][label] = met
            tables.append(pd.DataFrame({"group": g, "strategy": label, "month": cohorts["month"].astype(str),
                                        "n": [len(rr) for rr in members], "ret": ret6, "spy": spy_6m,
                                        "excess": exc}))
        res[g]["opportunistic_minus_routine"] = newey_west(excess[(g, "opportunistic")] - excess[(g, "routine")])

        rand = []
        for _ in range(draws):
            members = []
            for rr in cand:
                perm = rng.permutation(rr)
                _, fi = np.unique(sig["issuer_cik"].to_numpy()[perm], return_index=True)
                members.append(perm[np.sort(fi)[:top]])
            r = pd.Series(basket_returns(seg, members, cohorts, len(days), start, cost), index=days)
            rand.append(bt.metrics(r, rf, spy))
        rand = pd.DataFrame(rand)
        res[g][f"random_{top}"] = {k: {"median": float(rand[k].median()), "p05": float(rand[k].quantile(0.05)),
                                       "p95": float(rand[k].quantile(0.95))}
                                   for k in ("cagr", "sharpe", "sortino", "alpha_ann", "max_drawdown")}
        res[g]["random_sharpe_above_spy_share"] = float((rand["sharpe"] > res["SPY buy and hold"]["sharpe"]).mean())
        log.info("%s %s: done", first, g)
    for g in ORDER:
        if g != "cfo":
            res[g]["opportunistic_minus_cfo"] = newey_west(excess[(g, "opportunistic")] - excess[("cfo", "opportunistic")])
    res["period"] = [str(days[0].date()), str(days[-1].date())]
    res["formation_months"] = [str(cohorts["month"].iloc[0]), str(cohorts["month"].iloc[-1])]
    res["n_cohorts"] = len(cohorts)
    return res, pd.concat(tables)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-signals", action="store_true", help="reuse data/roles/signals.parquet")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--cost-bps", type=float, default=0.0, help="trading cost per side in basis points")
    ap.add_argument("--max-daily-move", type=float, default=10.0, help="drop windows with a larger one-day move")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    OUT.mkdir(parents=True, exist_ok=True)

    path = OUT / "signals.parquet"
    if args.reuse_signals:
        sig = pd.read_parquet(path)
    else:
        sig = build_signals()
        sig.to_parquet(path, index=False)
    cal = stock.market_calendar()
    seg, dropped = price_segments(sig, cal, args.max_daily_move)
    log.info("%d signals dropped for a one-day move above x%g", dropped, args.max_daily_move)
    rf = sources.fred_daily(["DTB3"])["DTB3"].dropna() / 100 / 252
    spy_adj = stock.prepare(sources.load_prices(stock.MARKET_ETF), cal)["adj_close"].ffill().to_numpy()

    counts = (sig.assign(priced=np.isfinite(seg).any(axis=1))
              .groupby(["group", "is_opportunistic"])["priced"].agg(["size", "sum"]))
    report = {"params": {"top": args.top, "draws": args.draws, "cost_bps_per_side": args.cost_bps, "hold_days": HOLD,
                         "min_price": MIN_PRICE, "groups": GROUPS, "max_daily_move": args.max_daily_move,
                         "dropped_for_daily_move": dropped},
              "signals": {f"{g}|{'opportunistic' if o else 'routine'}": {"signals": int(r["size"]),
                                                                          "with_entry_price": int(r["sum"])}
                          for (g, o), r in counts.iterrows()}}
    tables = []
    for name, first in PERIODS.items():
        res, tab = run_period(sig, seg, cal, first, args.top, args.draws, args.cost_bps / 1e4, rf, spy_adj)
        report[name] = res
        tables.append(tab.assign(period=name))
    tag = ("" if args.cost_bps == 0 else f"_cost{args.cost_bps:g}") + \
          ("" if args.max_daily_move == 10 else f"_move{args.max_daily_move:g}")
    (OUT / f"role_backtest{tag}.json").write_text(json.dumps(report, indent=2))
    pd.concat(tables).to_csv(OUT / f"cohorts{tag}.csv", index=False)

    for name in PERIODS:
        r = report[name]
        print(f"\n== {name}: {r['period'][0]} .. {r['period'][1]}, {r['n_cohorts']} cohorts")
        rows = {"SPY": r["SPY buy and hold"]}
        for g in ORDER:
            for label in ("opportunistic", "routine"):
                rows[f"{g} {label}"] = r[g][label]
        t = pd.DataFrame({k: {m: v.get(m) for m in ("cagr", "ann_vol", "sharpe", "max_drawdown", "beta", "alpha_ann")}
                          for k, v in rows.items()}).T
        t["excess_6m"] = [v["cohort_excess_6m"]["mean"] if "cohort_excess_6m" in v else np.nan for v in rows.values()]
        t["t"] = [v["cohort_excess_6m"]["t"] if "cohort_excess_6m" in v else np.nan for v in rows.values()]
        print(t.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
