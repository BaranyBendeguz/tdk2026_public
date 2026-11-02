# Opportunistic CFO purchases of large companies.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "modell" / "features"))
import backtest as bt  # noqa: E402
import data  # noqa: E402
import rolling  # noqa: E402
import sources  # noqa: E402
import stock  # noqa: E402
from robustness import newey_west  # noqa: E402

OUT = ROOT / "data" / "model" / "size"
MAX_AGE_DAYS = 400
MAX_TURNOVER = 0.5   # half of the company traded every day: the share count is wrong
MIN_TURNOVER = 1e-5  # 0.001% a day: the share count is wrong (MEI 2012: 37 billion shares in XBRL)
PRICE_TOLERANCE = 2.0
GROUPS = {"all": None, "cap >= 2bn": 2e9, "cap >= 10bn": 1e10, "S&P 500": "sp500"}
PERIODS = {"2006-2026": ("2006-01", "2026-12"), "2010-2026": ("2010-01", "2026-12"),
           "2012-2021": ("2012-01", "2021-12"), "2022-2026": ("2022-01", "2026-12")}
MODEL_PERIODS = {"2012-2026": ("2012-01", "2026-12"), "2012-2021": ("2012-01", "2021-12"),
                 "2022-2026": ("2022-01", "2026-12")}
RUNS = ["seed42", "seed1", "seed2", "seed3", "seed4", "ensemble"]
MIN_GROUP = 4


def split_factor(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache: dict) -> float:
    """Product of the Yahoo split ratios after `start` up to and including `end`."""
    if symbol not in cache:
        px = sources.load_prices(symbol)
        cache[symbol] = px["split"].replace(0, np.nan).dropna() if px is not None and "split" in px else pd.Series()
    s = cache[symbol]
    s = s[(s.index > start) & (s.index <= end)]
    return float(s.prod()) if len(s) else 1.0


def market_caps(sig: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    shares = sources.sec_shares_outstanding(sig["issuer_cik"].unique())
    shares = shares[shares["val"] > 0].copy()
    by_cik = dict(tuple(shares.groupby("cik")))
    anchor = stock.anchor_dates(sig["trans_date"], cal)
    price = np.exp(sig["stk_log_price"])
    ratio = price / sig["vwap"].where(sig["vwap"] > 0)
    off = (ratio > PRICE_TOLERANCE) | (ratio < 1 / PRICE_TOLERANCE)
    price = price.mask(off)
    print(f"market cap: {int(off.sum())} signals left out, the Yahoo close and the filed price disagree")
    splits: dict = {}
    out = pd.Series(np.nan, index=sig.index)
    for i, cik, sd, a, sym in zip(sig.index, sig["issuer_cik"], sig["signal_date"], anchor, sig["symbol"]):
        g = by_cik.get(int(cik))
        if g is None or pd.isna(a):
            continue
        g = g[(g["filed"] <= sd) & (g["end"] <= a) & (g["end"] >= a - pd.Timedelta(days=MAX_AGE_DAYS))]
        if g.empty:
            continue
        adj = [v * split_factor(sym, e, a, splits) for v, e in zip(g["val"], g["end"])]
        out[i] = float(np.median(adj)) * price[i]
    turnover = np.expm1(sig["stk_log_adv_1m"]) / out
    bad = (turnover > MAX_TURNOVER) | (turnover < MIN_TURNOVER)
    print(f"market cap: {int(out.notna().sum())} signals, {int(bad.sum())} dropped by the turnover check")
    return out.mask(bad)


def sp500_member(sig: pd.DataFrame) -> pd.Series:
    comp = sources.sp500_components()
    days = np.sort(comp["date"].unique())
    members = {d: set(g["ticker"].str.upper().str.replace(".", "-", regex=False))
               for d, g in comp.groupby("date")}
    out = pd.Series(False, index=sig.index)
    for i, sd, filed, sym in zip(sig.index, sig["signal_date"], sig["issuer_ticker"], sig["symbol"]):
        k = np.searchsorted(days, np.datetime64(sd), side="right") - 1
        if k < 0:
            continue
        cands = stock.clean_ticker(filed)
        if len(cands) > 1:
            cands = [c for c in cands if len(c) > 1]
        if not cands and isinstance(sym, str):
            cands = [sym]
        cands = set(cands)
        out[i] = bool(cands & members[days[k]])
    return out


def cohort_returns(sig: pd.DataFrame, cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig = sig.assign(month=sig["signal_date"].dt.to_period("M")).reset_index(drop=True)
    cohorts = bt.month_cohorts(sig, cal, bt.HOLD)
    sig = sig[sig["month"].isin(cohorts["month"])].reset_index(drop=True)
    paths = bt.price_paths(sig, cohorts, cal)
    x = sig["month"].map(cohorts.set_index("month")["x"]).to_numpy()
    sig["ret"] = paths[np.arange(len(sig)), x] - 1
    return sig, cohorts


def in_group(df: pd.DataFrame, rule) -> pd.Series:
    if rule is None:
        return pd.Series(True, index=df.index)
    if rule == "sp500":
        return df["sp500"]
    return df["market_cap"] >= rule


def by_period(x: pd.Series, periods: dict = PERIODS) -> dict:
    out = {}
    for name, (a, b) in periods.items():
        v = x[(x.index >= a) & (x.index <= b)].dropna()
        out[name] = newey_west(v.to_numpy(float)) if len(v) > 6 else None
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = stock.market_calendar()
    df, _, _ = data.load()
    df["market_cap"] = market_caps(df, cal)
    df["sp500"] = sp500_member(df)
    df[["event_id", "owner_cik", "issuer_cik", "symbol", "issuer_ticker", "signal_date", "market_cap", "sp500"]].to_parquet(
        OUT / "signal_size.parquet", index=False)
    yr = df["signal_date"].dt.year
    print("market cap coverage by year:", df.groupby(yr)["market_cap"].apply(lambda s: s.notna().mean()).round(2).to_dict())
    print("S&P 500 members by year:", df.groupby(yr)["sp500"].sum().to_dict())

    bench_adj = {b: stock.prepare(sources.load_prices(b), cal)["adj_close"].ffill().to_numpy() for b in ("SPY", "RSP")}

    # 1. model-free
    sig, cohorts = cohort_returns(df, cal)
    co = cohorts.assign(month=cohorts["month"].astype(str)).set_index("month")
    bench = pd.DataFrame({b: a[co["x"]] / a[co["e"]] - 1 for b, a in bench_adj.items()}, index=co.index)
    rows = []
    for gname, rule in GROUPS.items():
        g = sig[in_group(sig, rule) & np.isfinite(sig["ret"])].drop_duplicates(["month", "issuer_cik"])
        per = g.groupby(g["month"].astype(str))["ret"]
        mean, n = per.mean(), per.size()
        for b in bench:
            for period, nw in by_period((mean - bench[b]).dropna()).items():
                if nw:
                    m = n[(n.index >= PERIODS[period][0]) & (n.index <= PERIODS[period][1])]
                    rows.append({"group": gname, "benchmark": b, "period": period, "mean_excess_6m": nw["mean"],
                                 "t": nw["t"], "n_cohorts": nw["n_cohorts"], "signals": int(m.sum()),
                                 "signals_per_month_median": float(m.median())})
    free = pd.DataFrame(rows)
    free.to_csv(OUT / "model_free.csv", index=False)

    # 2. the rolling model inside each group
    size = df.set_index("event_id")[["market_cap", "sp500"]]
    rows = []
    for run in RUNS:
        pred = pd.read_parquet(rolling.OUT / run / "test_predictions.parquet").join(size, on="event_id")
        s, _ = cohort_returns(pred, cal)
        s = s.sort_values("pred", ascending=False).drop_duplicates(["month", "issuer_cik"])
        for gname, rule in GROUPS.items():
            g = s[in_group(s, rule) & np.isfinite(s["ret"])]
            res = {}
            for m, c in g.groupby(g["month"].astype(str)):
                if len(c) < MIN_GROUP:
                    continue
                top = c["pred"] >= c["pred"].median()
                res[m] = {"top_minus_bottom_mean": c.loc[top, "ret"].mean() - c.loc[~top, "ret"].mean(),
                          "top_minus_bottom_median": c.loc[top, "ret"].median() - c.loc[~top, "ret"].median(),
                          "top_minus_group": c.loc[top, "ret"].mean() - c["ret"].mean()}
            res = pd.DataFrame(res).T
            for col in res:
                for period, nw in by_period(res[col], MODEL_PERIODS).items():
                    if nw:
                        rows.append({"run": run, "group": gname, "spread": col, "period": period,
                                     "mean_6m": nw["mean"], "t": nw["t"], "n_cohorts": nw["n_cohorts"]})
    model = pd.DataFrame(rows)
    model.to_csv(OUT / "model_in_group.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(free.round(3).to_string(index=False), "\n")
    seeds = model[model["run"] != "ensemble"].groupby(["group", "spread", "period"]).agg(
        mean=("mean_6m", "median"), t_median=("t", "median"), t_min=("t", "min"), t_max=("t", "max"),
        n_cohorts=("n_cohorts", "median"))
    ens = model[model["run"] == "ensemble"].set_index(["group", "spread", "period"])[["mean_6m", "t"]].rename(
        columns={"mean_6m": "ens_mean", "t": "ens_t"})
    print(seeds.join(ens).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
