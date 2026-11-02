# Model data set: sample filters, one row per tradable signal, engineered features.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modell" / "features"))
import build_features as bf  # noqa: E402

FEATURES = ROOT / "data" / "features" / "cfo_opp_features.parquet"
TARGETS = ROOT / "data" / "features" / "cfo_opp_targets.parquet"
TARGET = "excess_ret_6m"
SIGNAL_KEY = ["owner_cik", "issuer_cik", "signal_date"]
MIN_PRICE = 1.0
CATEGORICAL = ["sector", "role_primary", "insider_type"]

ENGINEERED = {
    "ret_vs_mkt_3m": "stk_ret_3m - mkt_ret_3m",
    "ret_vs_mkt_6m": "stk_ret_6m - mkt_ret_6m",
    "ret_vs_mkt_12m": "stk_ret_12m - mkt_ret_12m",
    "mom_12_1": "12-month return without the last month: (1 + ret_12m) / (1 + ret_1m) - 1",
    "vol_adj_ret_3m": "stk_ret_3m / stk_vol_3m",
    "vol_adj_ret_12m": "stk_ret_12m / stk_vol_12m",
    "downside_share_3m": "stk_downside_vol_3m / stk_vol_3m (asymmetry of recent moves)",
    "vol_trend": "stk_vol_1m / stk_vol_12m (volatility rising or falling)",
    "cfo_share_of_insider_value_30d": "this purchase's value / all insider purchases of the stock in 30 days",
    "real_rate_3m": "macro_tbill_3m - macro_cpi_yoy",
    "signal_delay_days": "calendar days from trans_date to signal_date",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    e = pd.DataFrame(index=df.index)
    for h in ("3m", "6m", "12m"):
        e[f"ret_vs_mkt_{h}"] = df[f"stk_ret_{h}"] - df[f"mkt_ret_{h}"]
    e["mom_12_1"] = (1 + df["stk_ret_12m"]) / (1 + df["stk_ret_1m"]) - 1
    e["vol_adj_ret_3m"] = df["stk_ret_3m"] / df["stk_vol_3m"].where(df["stk_vol_3m"] > 0)
    e["vol_adj_ret_12m"] = df["stk_ret_12m"] / df["stk_vol_12m"].where(df["stk_vol_12m"] > 0)
    e["downside_share_3m"] = df["stk_downside_vol_3m"] / df["stk_vol_3m"].where(df["stk_vol_3m"] > 0)
    e["vol_trend"] = df["stk_vol_1m"] / df["stk_vol_12m"].where(df["stk_vol_12m"] > 0)
    e["cfo_share_of_insider_value_30d"] = np.expm1(df["log_value"]) / np.expm1(df["ins_log_value_30d"]).where(
        df["ins_log_value_30d"] > 0)
    e["real_rate_3m"] = df["macro_tbill_3m"] - df["macro_cpi_yoy"]
    e["signal_delay_days"] = (df["signal_date"] - df["trans_date"]).dt.days
    return e.replace([np.inf, -np.inf], np.nan)


def to_signals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (owner, issuer, signal_date): the latest trade day, with value and fills of all days."""
    df = df.sort_values(SIGNAL_KEY + ["trans_date"])
    g = df.groupby(SIGNAL_KEY, sort=False)
    total_value = g["value_usd"].transform(lambda s: s.sum(min_count=1))
    total_fills = g["n_fills"].transform("sum")
    n_days = g["trans_date"].transform("size")
    df = df.assign(log_value=np.log1p(total_value), n_fills=total_fills, n_trade_days=n_days)
    return df.drop_duplicates(SIGNAL_KEY, keep="last")


def load(min_price: float = MIN_PRICE) -> tuple[pd.DataFrame, list[str], dict]:
    """Returns (data, feature_names, sample_counts). data holds meta columns, the features and TARGET."""
    df = pd.read_parquet(FEATURES).merge(pd.read_parquet(TARGETS), on="event_id")
    counts = {"events": len(df)}
    steps = [
        ("has_symbol", df["symbol"].notna()),
        ("has_target", df[TARGET].notna()),
        ("stock_purchase", df["only_deriv"] == 0),
        (f"price_at_least_{min_price:g}", np.exp(df["stk_log_price"]) >= min_price),
    ]
    keep = pd.Series(True, index=df.index)
    for name, m in steps:
        keep &= m.fillna(False)
        counts[name] = int(keep.sum())
    df = to_signals(df[keep])
    counts["signals"] = len(df)
    df = pd.concat([df, engineer(df)], axis=1)
    features = bf.FEATURES + bf.EXTRA_FEATURES + ["n_trade_days"] + list(ENGINEERED)
    return df.sort_values("signal_date").reset_index(drop=True), features, counts
