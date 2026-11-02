# Checks on the feature table.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_features as bf  # noqa: E402
import sources  # noqa: E402
import stock  # noqa: E402

CHECKED = [f for f in stock.STOCK_FEATURES if f not in ("stk_log_price", "stk_days_since_price")]


def point_in_time(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    sample = df[df["symbol"].notna() & df["stk_ret_1m"].notna()].sample(n, random_state=seed)
    spy = sources.load_prices(stock.MARKET_ETF)
    rows = []
    for r in sample.itertuples(index=False):
        anchor = stock.anchor_dates(pd.Series([r.trans_date]), spy.index).iloc[0]
        cal = spy.index[spy.index <= anchor]
        spy_cut = stock.prepare(spy[spy.index <= anchor], cal)
        px = sources.load_prices(r.symbol)
        p = stock.prepare(px[px.index <= anchor], cal)
        f = stock.stock_daily(p, np.log(spy_cut["adj_close"]).diff()).loc[anchor, CHECKED]
        stored = pd.Series({c: getattr(r, c) for c in CHECKED})
        diff = (f.astype(float) - stored.astype(float)).abs() / (stored.astype(float).abs() + 1e-12)
        rows.append({"event_id": r.event_id, "symbol": r.symbol, "max_rel_diff": float(np.nanmax(diff)),
                     "worst": diff.idxmax() if diff.notna().any() else None})
    return pd.DataFrame(rows)


def ranges(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in bf.FEATURES + bf.EXTRA_FEATURES if pd.api.types.is_numeric_dtype(df[c])]
    q = df[num].quantile([0.01, 0.5, 0.99]).T
    q.columns = ["p01", "p50", "p99"]
    q.insert(0, "missing", df[num].isna().mean())
    return q


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=bf.OUT_DIR / "cfo_opp_features.parquet")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args(argv)
    df = pd.read_parquet(args.features)

    pit = point_in_time(df, args.n)
    bad = pit[pit["max_rel_diff"] > 1e-6]
    print(f"point in time: {len(pit)} events recomputed from history cut at the anchor; "
          f"{len(bad)} differ (max relative difference {pit['max_rel_diff'].max():.2e})")
    if len(bad):
        print(bad.head(20).to_string(index=False))
    pd.set_option("display.width", 200)
    print(ranges(df).round(4).to_string())
    return int(len(bad) > 0)


if __name__ == "__main__":
    sys.exit(main())
