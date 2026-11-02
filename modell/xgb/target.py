# Six-month excess return target (over SPY) for each signal.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modell" / "features"))
import sources  # noqa: E402
import stock  # noqa: E402

FEATURES = ROOT / "data" / "features" / "cfo_opp_features.parquet"
TARGETS = ROOT / "data" / "features" / "cfo_opp_targets.parquet"
HORIZON = 126  # market days, about 6 months


def compute_targets(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    cal = stock.market_calendar()
    pos = cal.searchsorted(df["signal_date"].to_numpy(), side="right")
    ok = pos + horizon < len(cal)
    out = pd.DataFrame({"event_id": df["event_id"].to_numpy()})
    out["entry_date"] = pd.NaT
    out["exit_date"] = pd.NaT
    out.loc[pos < len(cal), "entry_date"] = cal[pos[pos < len(cal)]]
    out.loc[ok, "exit_date"] = cal[pos[ok] + horizon]

    def leg(adj: pd.Series, idx) -> tuple[np.ndarray, np.ndarray]:
        e = adj.reindex(out.loc[idx, "entry_date"]).to_numpy()
        x = adj.reindex(out.loc[idx, "exit_date"]).to_numpy()
        return e, x

    spy = stock.prepare(sources.load_prices(stock.MARKET_ETF), cal)["adj_close"].ffill(limit=stock.FFILL_LIMIT)
    e, x = leg(spy, out.index)
    out["spy_ret"] = x / e - 1
    out["stock_ret"] = np.nan
    out["max_daily_move"] = np.nan
    sym = df["symbol"].to_numpy()
    for s in pd.unique(sym[pd.notna(sym)]):
        idx = out.index[sym == s]
        adj = stock.prepare(sources.load_prices(s), cal)["adj_close"].ffill(limit=stock.FFILL_LIMIT)
        e, x = leg(adj, idx)
        out.loc[idx, "stock_ret"] = x / e - 1
        # largest one-day price move in the holding window: x5 or /5 in a day is almost always a data error
        # (an unrecorded split, a misplaced decimal)
        move = np.log(adj).diff().abs()
        for i, (a, b) in zip(idx, out.loc[idx, ["entry_date", "exit_date"]].itertuples(index=False)):
            if pd.notna(b):
                out.at[i, "max_daily_move"] = np.exp(move.loc[a:b].iloc[1:].max())
    out["excess_ret_6m"] = out["stock_ret"] - out["spy_ret"]
    return out


def main() -> int:
    df = pd.read_parquet(FEATURES, columns=["event_id", "signal_date", "symbol"])
    t = compute_targets(df)
    t.to_parquet(TARGETS, index=False)
    s = t["excess_ret_6m"]
    print(f"{s.notna().sum()} of {len(t)} events have a realised target; "
          f"mean {s.mean():.4f}, median {s.median():.4f}, std {s.std():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
