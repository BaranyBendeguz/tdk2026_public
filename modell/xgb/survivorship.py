# Sensitivity of the CFO results to the survivorship bias of the Yahoo price data.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

FEATURES = ROOT / "data" / "features" / "cfo_opp_features.parquet"
TARGETS = ROOT / "data" / "features" / "cfo_opp_targets.parquet"
TRANSACTIONS = ROOT / "data" / "parquet" / "insider_transactions"
SEC_TICKERS = ROOT / "data" / "market" / "sec_company_tickers.json"
OUT = ROOT / "data" / "model" / "survivorship.json"
LATER = pd.Timedelta(days=730)
TEST_START = pd.Timestamp("2022-01-01")


def last_filing_by_issuer() -> pd.Series:
    parts = [pd.read_parquet(p, columns=["issuer_cik", "filing_date"]) for p in sorted(TRANSACTIONS.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    return pd.to_datetime(df["filing_date"]).groupby(df["issuer_cik"]).max()


def main() -> int:
    ev = pd.read_parquet(FEATURES).merge(pd.read_parquet(TARGETS), on="event_id", how="left")
    # only events whose 6-month window could have closed: up to the last signal with a realised target
    ev = ev[(ev["only_deriv"] == 0) & (ev["signal_date"] <= ev.loc[ev[data.TARGET].notna(), "signal_date"].max())].copy()
    ev["covered"] = ev["symbol"].notna() & ev[data.TARGET].notna()
    sec = {int(v["cik_str"]) for v in json.loads(SEC_TICKERS.read_text()).values()}
    ev["in_sec_map"] = ev["issuer_cik"].astype(int).isin(sec)
    last = last_filing_by_issuer()
    ev["files_later"] = ev["issuer_cik"].map(last) >= ev["signal_date"] + LATER
    ev["year"] = ev["signal_date"].dt.year
    ev["alive"] = ev["in_sec_map"] | ev["files_later"]

    by_year = ev.groupby("year").agg(
        events=("covered", "size"), coverage=("covered", "mean"), alive_share=("alive", "mean"),
        in_sec_map=("in_sec_map", "mean"),
        coverage_alive=("covered", lambda s: s[ev.loc[s.index, "alive"]].mean()),
        coverage_not_alive=("covered", lambda s: s[~ev.loc[s.index, "alive"]].mean()))

    def profile(m: pd.Series) -> dict:
        x = ev[m]
        return {"events": int(len(x)), "median_trade_price": float(x["vwap"].median()),
                "share_trade_price_below_5": float((x["vwap"] < 5).mean()),
                "median_trade_value": float(x["value_usd"].median()), "alive_share": float(x["alive"].mean()),
                "in_sec_map_share": float(x["in_sec_map"].mean())}

    test = ev["signal_date"] >= TEST_START
    profiles = {f"{p}_{c}": profile(pm & (ev["covered"] == cv_))
                for p, pm in (("all", pd.Series(True, index=ev.index)), ("test", test))
                for c, cv_ in (("covered", True), ("missing", False))}

    sens = {}
    for name, m in (("all", pd.Series(True, index=ev.index)), ("test", test)):
        cov = ev.loc[m & ev["covered"], data.TARGET]
        share = float(ev.loc[m, "covered"].mean())
        sens[name] = {"coverage": share, "covered_mean_excess": float(cov.mean()),
                      **{f"if_missing_{q:+.0%}": float(share * cov.mean() + (1 - share) * q)
                         for q in (0.0, -0.25, -0.5, -1.0)}}

    out = {"by_year": by_year.round(4).reset_index().to_dict(orient="records"), "profiles": profiles,
           "baseline_sensitivity": sens}
    OUT.write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 200)
    print(by_year.round(3).to_string())
    print(pd.DataFrame(profiles).T.round(3).to_string())
    print(pd.DataFrame(sens).T.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
