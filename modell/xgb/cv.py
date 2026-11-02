# Walk-forward validation with purging, and the evaluation metrics.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TEST_START = pd.Timestamp("2022-01-01")
VAL_YEARS = list(range(2013, 2022))
WINSOR_Q = (0.01, 0.99)
MIN_MONTH_N = 10  # months with fewer signals give no usable rank correlation


def _after(df: pd.DataFrame, train_start: pd.Timestamp | None) -> pd.Series:
    return df["signal_date"] >= train_start if train_start is not None else pd.Series(True, index=df.index)


def folds(df: pd.DataFrame, val_years=VAL_YEARS,
          train_start: pd.Timestamp | None = None) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(year, train positions, validation positions); training signals start at train_start if given."""
    out = []
    for y in val_years:
        start, end = pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1)
        train = np.flatnonzero(((df["exit_date"] < start) & _after(df, train_start)).to_numpy())
        val = np.flatnonzero(((df["signal_date"] >= start) & (df["signal_date"] < end)).to_numpy())
        out.append((y, train, val))
    return out


def halfyear_folds(df: pd.DataFrame, first_year: int, last_year: int,
                   train_start: pd.Timestamp | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Like folds(), with one validation fold per half year (2018H1, 2018H2, ...)."""
    out = []
    for y in range(first_year, last_year + 1):
        for h, (a, b) in (("H1", (1, 7)), ("H2", (7, 13))):
            start = pd.Timestamp(y, a, 1)
            end = pd.Timestamp(y + (b == 13), 1 if b == 13 else b, 1)
            train = np.flatnonzero(((df["exit_date"] < start) & _after(df, train_start)).to_numpy())
            val = np.flatnonzero(((df["signal_date"] >= start) & (df["signal_date"] < end)).to_numpy())
            out.append((f"{y}{h}", train, val))
    return out


def test_split(df: pd.DataFrame, start: pd.Timestamp = TEST_START,
               train_start: pd.Timestamp | None = None) -> tuple[np.ndarray, np.ndarray]:
    train = np.flatnonzero(((df["exit_date"] < start) & _after(df, train_start)).to_numpy())
    test = np.flatnonzero((df["signal_date"] >= start).to_numpy())
    return train, test


def winsor_bounds(y: pd.Series) -> tuple[float, float]:
    """Clipping bounds of the training target (computed on the training fold only)."""
    lo, hi = y.quantile(list(WINSOR_Q))
    return float(lo), float(hi)


def ic(pred, y) -> float:
    """Spearman rank correlation between prediction and realised excess return (information coefficient)."""
    r = spearmanr(pred, y).statistic
    return float(r) if np.isfinite(r) else 0.0


def ic_by_month(pred, y, dates, min_n: int = MIN_MONTH_N) -> np.ndarray:
    """Spearman IC inside each calendar month with at least min_n signals."""
    d = pd.DataFrame({"p": np.asarray(pred, float), "y": np.asarray(y, float),
                      "m": pd.to_datetime(np.asarray(dates)).to_period("M")})
    return np.array([ic(g["p"], g["y"]) for _, g in d.groupby("m") if len(g) >= min_n])


def ic_within_month(pred, y, dates, min_n: int = MIN_MONTH_N) -> float:
    """Mean of the monthly ICs: how well the model ranks the signals of the same month, which is what a
    monthly top-N strategy uses. Market timing does not count here."""
    v = ic_by_month(pred, y, dates, min_n)
    return float(v.mean()) if len(v) else 0.0


def evaluate(pred: np.ndarray, y: pd.Series, dates: pd.Series, bounds: tuple[float, float],
             train_mean: float) -> dict:
    """Prediction quality and the return of the buys the model ranks best."""
    y = pd.Series(np.asarray(y, float))
    pred = pd.Series(np.asarray(pred, float))
    yc = y.clip(*bounds)
    q = pd.qcut(pred.rank(method="first"), 5, labels=False)
    quarter = pd.Series(pd.PeriodIndex(pd.to_datetime(dates.to_numpy()), freq="Q"))
    by_q = [ic(pred[m], y[m]) for _, m in quarter.groupby(quarter).groups.items() if len(m) >= 20]
    by_q = [v for v in by_q if np.isfinite(v)]
    top, bottom = y[q == 4], y[q == 0]
    by_m = ic_by_month(pred, y, dates)
    return {
        "n": int(len(y)),
        "ic": ic(pred, y),
        "ic_within_month_mean": float(by_m.mean()) if len(by_m) else np.nan,
        "ic_within_month_pos_share": float((by_m > 0).mean()) if len(by_m) else np.nan,
        "ic_quarterly_mean": float(np.mean(by_q)) if by_q else np.nan,
        "ic_quarterly_pos_share": float(np.mean(np.array(by_q) > 0)) if by_q else np.nan,
        "rmse_winsorized": float(np.sqrt(np.mean((pred - yc) ** 2))),
        "r2_oos_winsorized": float(1 - np.sum((yc - pred) ** 2) / np.sum((yc - train_mean) ** 2)),
        "mean_excess_all": float(y.mean()),
        "median_excess_all": float(y.median()),
        "mean_excess_top_quintile": float(top.mean()),
        "median_excess_top_quintile": float(top.median()),
        "mean_excess_bottom_quintile": float(bottom.mean()),
        "top_minus_bottom": float(top.mean() - bottom.mean()),
        "hit_rate_top_quintile": float((top > 0).mean()),
        "hit_rate_all": float((y > 0).mean()),
        "mean_excess_by_quintile": [float(y[q == k].mean()) for k in range(5)],
    }
