# Factor-model alphas and event-time entry for the opportunistic CFO purchases.

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

OUT = ROOT / "data" / "model" / "factor_event"
SIZE = ROOT / "data" / "model" / "size" / "signal_size.parquet"
MODELS = {"CAPM": ["mkt_rf"], "FF3": ["mkt_rf", "smb", "hml"], "Carhart": ["mkt_rf", "smb", "hml", "mom"],
          "FF5+Mom": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]}
PERIODS = {"2006-2026": ("2006-01-01", "2026-12-31"), "2012-2021": ("2012-01-01", "2021-12-31"),
           "2022-2026": ("2022-01-01", "2026-12-31")}
NW_DAILY = 21
HOLDS = (21, 63, 126)
CAR_DAYS = (1, 5, 21, 63, 126)
RUNS = ["seed42", "seed1", "seed2", "seed3", "seed4", "ensemble"]


def nw_ols(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS coefficients, Newey-West standard errors and R2; x without the constant."""
    X = np.column_stack([np.ones(len(y)), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    s = u.T @ u
    for k in range(1, lags + 1):
        g = u[k:].T @ u[:-k]
        s += (1 - k / (lags + 1)) * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ s @ xtx_inv))
    return b, se, float(1 - e.var() / y.var())


def factor_table(name: str, r: pd.Series, ff: pd.DataFrame) -> list[dict]:
    rows = []
    d = pd.concat([r.rename("r"), ff], axis=1, join="inner").dropna()
    for period, (a, b) in PERIODS.items():
        p = d.loc[a:b]
        if len(p) < 250:
            continue
        y = (p["r"] - p["rf"]).to_numpy()
        for model, cols in MODELS.items():
            coef, se, r2 = nw_ols(y, p[cols].to_numpy(), NW_DAILY)
            rows.append({"portfolio": name, "period": period, "model": model, "alpha_ann": coef[0] * 252,
                         "t_alpha": coef[0] / se[0], "r2": r2, "n_days": len(p),
                         **{f"b_{c}": v for c, v in zip(cols, coef[1:])}})
    return rows


class Prices:
    """Adjusted closes of every symbol on the market calendar (forward-filled as in backtest.py)."""

    def __init__(self, cal: pd.DatetimeIndex):
        self.cal, self.cache = cal, {}

    def __call__(self, sym: str) -> np.ndarray:
        if sym not in self.cache:
            px = sources.load_prices(sym)
            self.cache[sym] = (stock.prepare(px, self.cal)["adj_close"].ffill(limit=stock.FFILL_LIMIT).to_numpy()
                               if px is not None else np.full(len(self.cal), np.nan))
        return self.cache[sym]


def cohort_daily(sig: pd.DataFrame, score: str | None, top: int | None, cal: pd.DatetimeIndex) -> pd.Series:
    """Daily returns of the monthly-cohort portfolio of backtest.py: per month the top `top` distinct issuers by
    `score` (all if top is None)."""
    sig = sig.assign(month=sig["signal_date"].dt.to_period("M")).reset_index(drop=True)
    cohorts = bt.month_cohorts(sig, cal, bt.HOLD)
    sig = sig[sig["month"].isin(cohorts["month"])].reset_index(drop=True)
    paths = bt.price_paths(sig, cohorts, cal)
    has = np.isfinite(paths).any(axis=1)
    members = []
    for m in cohorts["month"]:
        s = sig[(sig["month"] == m) & has]
        if score is not None:
            s = s.sort_values(score, ascending=False)
        s = s.drop_duplicates("issuer_cik")
        members.append(s.index.to_numpy()[:top] if top else s.index.to_numpy())
    first, last = cohorts["e"].iloc[0], cohorts["x"].iloc[-1]
    r = bt.portfolio(bt.cohort_returns(paths, members, cohorts, 0.0))[first + 1:last + 1]
    return pd.Series(r, index=cal[first + 1:last + 1]).dropna()


def event_daily(sig: pd.DataFrame, hold: int, prices: Prices, cal: pd.DatetimeIndex,
                buy_and_hold: bool = False) -> pd.Series:
    """Calendar-time portfolio: each signal held from the close after its signal_date for `hold` days. Equal weight
    every day, or (buy_and_hold) weighted by the position's value since entry."""
    total, weight = np.zeros(len(cal)), np.zeros(len(cal))
    entry = cal.searchsorted(sig["signal_date"].to_numpy(), side="right")
    for sym, e in zip(sig["symbol"], entry):
        x = e + hold
        if x >= len(cal):
            continue
        adj = prices(sym)
        r = adj[e + 1:x + 1] / adj[e:x] - 1
        ok = np.isfinite(r)
        if buy_and_hold:
            w = np.nan_to_num(np.r_[1.0, np.cumprod(1 + np.where(ok, r, 0.0))[:-1]] * np.isfinite(adj[e]))
        else:
            w = np.ones(len(r))
        total[e + 1:x + 1][ok] += (w * r)[ok]
        weight[e + 1:x + 1][ok] += w[ok]
    with np.errstate(invalid="ignore"):
        out = total / weight
    return pd.Series(out, index=cal).where(weight > 0).dropna()


def trailing_flag(sig: pd.DataFrame, col: str, q: float, above: bool) -> pd.Series:
    """True when sig[col] is beyond the q-quantile of the signals of the previous 365 days (point in time)."""
    s = sig.sort_values("signal_date")
    out = pd.Series(False, index=sig.index)
    dates, vals = s["signal_date"].to_numpy(), s[col].to_numpy(float)
    for i, (d, v) in zip(s.index, zip(dates, vals)):
        w = vals[(dates < d) & (dates >= d - np.timedelta64(365, "D"))]
        w = w[np.isfinite(w)]
        if len(w) >= 50 and np.isfinite(v):
            th = np.quantile(w, q)
            out[i] = v >= th if above else v <= th
    return out


def car_table(sig: pd.DataFrame, prices: Prices, cal: pd.DatetimeIndex, spy: np.ndarray) -> pd.DataFrame:
    anchor = cal.searchsorted(sig["trans_date"].to_numpy(), side="right") - 1
    entry = cal.searchsorted(sig["signal_date"].to_numpy(), side="right")
    rows = []
    for sym, a, e, sd in zip(sig["symbol"], anchor, entry, sig["signal_date"]):
        if e + max(CAR_DAYS) >= len(cal) or a < 0:
            continue
        adj = prices(sym)
        row = {"month": sd.to_period("M"), "trade_to_entry": adj[e] / adj[a] - spy[e] / spy[a],
               "delay_days": e - a}
        for h in CAR_DAYS:
            row[f"entry_to_{h}d"] = adj[e + h] / adj[e] - spy[e + h] / spy[e]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = stock.market_calendar()
    ff = sources.ff_factors_daily()
    prices = Prices(cal)
    spy = prices(stock.MARKET_ETF)
    df, _, _ = data.load()
    size = pd.read_parquet(SIZE)[["event_id", "market_cap", "sp500"]]
    df = df.merge(size, on="event_id", how="left")

    series = {}
    # monthly cohorts, all CFO signals and size groups, 2006-2026
    series["cohort: all CFO buys"] = cohort_daily(df, None, None, cal)
    series["cohort: cap >= 2bn"] = cohort_daily(df[df["market_cap"] >= 2e9], None, None, cal)
    series["cohort: S&P 500"] = cohort_daily(df[df["sp500"].fillna(False)], None, None, cal)
    # monthly cohorts from the rolling predictions, 2012-2026
    for run in RUNS:
        pred = pd.read_parquet(rolling.OUT / run / "test_predictions.parquet")
        for top in (5, 10, 20):
            series[f"cohort: model top {top} ({run})"] = cohort_daily(pred, "pred", top, cal)
    pred = pd.read_parquet(rolling.OUT / "ensemble" / "test_predictions.parquet").merge(
        df[["event_id", "stk_vol_3m"]], on="event_id")
    series["cohort: lowest volatility 5 (2012-)"] = cohort_daily(pred.assign(nv=-pred["stk_vol_3m"]), "nv", 5, cal)
    series["cohort: all CFO buys (2012-)"] = cohort_daily(pred, None, None, cal)

    # event time
    for h in HOLDS:
        series[f"event: all CFO buys, hold {h}d"] = event_daily(df, h, prices, cal)
        series[f"event BH: all CFO buys, hold {h}d"] = event_daily(df, h, prices, cal, buy_and_hold=True)
    cap = df["market_cap"]
    groups = {"micro < 300m": cap < 3e8, "small 300m-2bn": (cap >= 3e8) & (cap < 2e9), "cap >= 2bn": cap >= 2e9,
              "no market cap, 2010-": cap.isna() & (df["signal_date"] >= "2010-01-01")}
    for g, m in groups.items():
        for h in (21, 126):
            series[f"event: {g}, hold {h}d"] = event_daily(df[m], h, prices, cal)
            series[f"event BH: {g}, hold {h}d"] = event_daily(df[m], h, prices, cal, buy_and_hold=True)
    pred["model_q5"] = trailing_flag(pred, "pred", 0.8, above=True)
    pred["lowvol_q1"] = trailing_flag(pred, "stk_vol_3m", 0.2, above=False)
    for h in HOLDS:
        series[f"event: model top quintile, hold {h}d"] = event_daily(pred[pred["model_q5"]], h, prices, cal)
        series[f"event: low-volatility quintile, hold {h}d"] = event_daily(pred[pred["lowvol_q1"]], h, prices, cal)
        series[f"event: all CFO buys (2012-), hold {h}d"] = event_daily(pred, h, prices, cal)
    series["SPY"] = pd.Series(spy[1:] / spy[:-1] - 1, index=cal[1:])

    pd.DataFrame(series).to_parquet(OUT / "daily_returns.parquet")
    rows = []
    for name, r in series.items():
        rows += factor_table(name, r, ff)
    alphas = pd.DataFrame(rows)
    hold = alphas["portfolio"].str.extract(r"hold (\d+)d")[0].astype(float)
    alphas["breakeven_round_trip"] = alphas["alpha_ann"] / (252 / hold)
    alphas.to_csv(OUT / "alphas.csv", index=False)

    car = car_table(df, prices, cal, spy)
    cols = ["trade_to_entry"] + [f"entry_to_{h}d" for h in CAR_DAYS]
    car_rows = []
    for period, (a, b) in PERIODS.items():
        c = car[(car["month"] >= pd.Period(a, "M")) & (car["month"] <= pd.Period(b, "M"))]
        monthly = c.groupby("month")[cols].mean()
        for col in cols:
            nw = newey_west(monthly[col].to_numpy(float))
            car_rows.append({"period": period, "window": col, "mean_signal": float(c[col].mean()),
                             "median_signal": float(c[col].median()), "mean_of_months": nw["mean"], "t": nw["t"],
                             "n_signals": int(c[col].notna().sum()), "n_months": nw["n_cohorts"]})
    car_df = pd.DataFrame(car_rows)
    car_df.to_csv(OUT / "car.csv", index=False)
    print("filing delay (market days from trade to entry): median", float(car["delay_days"].median()))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    main_rows = alphas[~alphas["portfolio"].str.contains(r"\(seed")]
    print(main_rows.pivot_table(index=["portfolio", "period"], columns="model", values=["alpha_ann", "t_alpha"])
          .round(3).to_string())
    seeds = alphas[alphas["portfolio"].str.contains(r"\(seed")].copy()
    seeds["portfolio"] = seeds["portfolio"].str.replace(r" \(seed\d+\)", "", regex=True)
    print(seeds.groupby(["portfolio", "period", "model"])[["alpha_ann", "t_alpha"]].median().unstack("model")
          .round(3).to_string())
    print(car_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
