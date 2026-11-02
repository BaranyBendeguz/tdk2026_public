# Macro features: the latest value already published before the trade date.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import sources

DAILY = {
    "macro_fed_funds": "DFF",
    "macro_tbill_3m": "DTB3",
    "macro_treasury_10y": "DGS10",
    "macro_curve_10y2y": "T10Y2Y",
    "macro_credit_spread": "BAA10Y",  # Moody's Baa yield minus the 10-year Treasury yield
    "macro_vix": "VIXCLS",
}
VINTAGE = {"macro_unemployment": "UNRATE", "macro_cpi_yoy": "CPIAUCSL", "macro_gdp_growth": "GDPC1"}
MACRO_FEATURES = list(DAILY) + list(VINTAGE)


def _from_vintages(v: pd.DataFrame, feature: str) -> pd.Series:
    """One value per vintage_date from the last observations of that vintage."""
    v = v.dropna(subset=["obs_date"]).sort_values(["vintage_date", "obs_date"])
    out = {}
    for vintage, g in v.groupby("vintage_date"):
        s = g.set_index("obs_date")["value"]
        last = s.index[-1]
        if feature == "macro_unemployment":
            out[vintage] = s.iloc[-1]
        elif feature == "macro_cpi_yoy":
            base = s.get(last - pd.DateOffset(years=1))
            out[vintage] = (s.iloc[-1] / base - 1) * 100 if base else np.nan
        else:  # real GDP growth, quarter on quarter, annualised (BEA convention)
            out[vintage] = ((s.iloc[-1] / s.iloc[-2]) ** 4 - 1) * 100 if len(s) > 1 else np.nan
    return pd.Series(out, dtype=float)


def macro_features(events: pd.DataFrame) -> pd.DataFrame:
    """Indexed by event_id."""
    asof = events["trans_date"] - pd.Timedelta(days=1)
    out = pd.DataFrame(index=events["event_id"])
    daily = sources.fred_daily(list(DAILY.values()))
    for name, series in DAILY.items():
        s = daily[series].dropna()
        pos = s.index.searchsorted(asof.to_numpy(), side="right") - 1
        out[name] = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    with ThreadPoolExecutor(len(VINTAGE)) as ex:
        vintages = dict(zip(VINTAGE, ex.map(lambda s: sources.alfred_vintages(s, asof.unique()), VINTAGE.values())))
    for name, v in vintages.items():
        out[name] = asof.map(_from_vintages(v, name)).to_numpy()
    return out
