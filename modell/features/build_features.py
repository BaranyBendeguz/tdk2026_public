# Feature table for the opportunistic CFO purchases.

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import insider  # noqa: E402
import macro  # noqa: E402
import sources  # noqa: E402
import stock  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OPP_DIR = ROOT / "data" / "opportunistic_buys"
OUT_DIR = ROOT / "data" / "features"
EVENT_KEY = ["owner_cik", "issuer_cik", "trans_date"]
ROW_COLS = ["accession_number", "filing_date", "issuer_cik", "issuer_name", "issuer_ticker", "owner_cik",
            "owner_name", "is_director", "is_officer", "is_ten_pct_owner", "is_other", "officer_title", "table",
            "security_title", "trans_date", "trans_shares", "trans_price", "shares_owned_after", "direct_indirect",
            "nature_of_ownership", "filing_lag_days"]

TRADE_FEATURES = ["log_value", "n_fills", "price_vs_close", "trade_value_to_adv"]
EXTRA_FEATURES = ["x_frac_holdings_bought", "x_filing_lag_days"]
FEATURES = (stock.STOCK_FEATURES + stock.SECTOR_FEATURES + macro.MACRO_FEATURES + insider.ACTIVITY_FEATURES
            + insider.ROLE_FEATURES + TRADE_FEATURES)
META = ["event_id", "owner_cik", "issuer_cik", "trans_date", "signal_date", "owner_name", "issuer_name",
        "issuer_ticker", "officer_title", "symbol", "symbol_source", "symbol_status", "value_usd", "vwap", "shares",
        "n_rows", "only_deriv", "has_price"]

log = logging.getLogger("features")


def build_events(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (owner, issuer, trans_date)."""
    for c in ("trans_date", "filing_date"):
        rows[c] = pd.to_datetime(rows[c])
    # the same line filed in two accessions (a Form 5 repeating a Form 4) counts once
    rows = rows.drop_duplicates(EVENT_KEY + ["table", "security_title", "trans_shares", "trans_price",
                                             "shares_owned_after", "direct_indirect", "nature_of_ownership"])
    t1 = rows["table"].eq("nonderiv")
    priced = t1 & (rows["trans_price"] > 0) & (rows["trans_shares"] > 0)
    direct = t1 & rows["direct_indirect"].eq("D")
    rows = rows.assign(
        is_t1=t1.astype(int),
        t1_shares=rows["trans_shares"].where(t1),
        priced_value=(rows["trans_shares"] * rows["trans_price"]).where(priced),
        priced_shares=rows["trans_shares"].where(priced),
        direct_bought=rows["trans_shares"].where(direct),
        direct_after=rows["shares_owned_after"].where(direct),
    )
    g = rows.groupby(EVENT_KEY)
    ev = g.agg(
        signal_date=("filing_date", "min"),
        owner_name=("owner_name", "first"),
        issuer_name=("issuer_name", "first"),
        officer_title=("officer_title", "first"),
        is_director=("is_director", "max"),
        is_officer=("is_officer", "max"),
        is_ten_pct_owner=("is_ten_pct_owner", "max"),
        is_other=("is_other", "max"),
        n_rows=("table", "size"),
        n_fills=("is_t1", "sum"),
        shares=("t1_shares", "sum"),
        value_usd=("priced_value", lambda s: s.sum(min_count=1)),
        priced_shares=("priced_shares", "sum"),
        direct_bought=("direct_bought", "sum"),
        direct_after=("direct_after", "max"),
        x_filing_lag_days=("filing_lag_days", "min"),
    )
    ev["issuer_ticker"] = g["issuer_ticker"].agg(lambda s: "|".join(dict.fromkeys(s.dropna().astype(str))))
    ev["vwap"] = ev["value_usd"] / ev["priced_shares"].where(ev["priced_shares"] > 0)
    ev["has_price"] = ev["vwap"].notna().astype(int)
    ev["only_deriv"] = (ev["n_fills"] == 0).astype(int)
    ev["x_frac_holdings_bought"] = (ev["direct_bought"] / ev["direct_after"].where(ev["direct_after"] > 0)).clip(0, 1)
    ev = ev.drop(columns=["priced_shares", "direct_bought", "direct_after"]).reset_index()
    ev.insert(0, "event_id", np.arange(len(ev)))
    return ev


def coverage_report(df: pd.DataFrame) -> dict:
    nonnull = {c: round(float(df[c].notna().mean()), 4) for c in FEATURES + EXTRA_FEATURES}
    matched = df["symbol"].notna()
    by_year = df.groupby(df["signal_date"].dt.year)["symbol"].apply(lambda s: round(float(s.notna().mean()), 3))
    return {
        "events": len(df),
        "symbol_status": df["symbol_status"].value_counts().to_dict(),
        "symbol_matched_share": round(float(matched.mean()), 4),
        "symbol_matched_share_per_signal_year": {int(k): v for k, v in by_year.items()},
        "feature_nonnull_share_all_events": nonnull,
        "feature_nonnull_share_matched_events": {c: round(float(df.loc[matched, c].notna().mean()), 4)
                                                 for c in FEATURES + EXTRA_FEATURES},
        "sector": df["sector"].value_counts(dropna=False).to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opp", type=Path, default=OPP_DIR / "cfo_opp.csv")
    ap.add_argument("--transactions", type=Path, default=OPP_DIR / "insider_transactions.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--refresh-sec", action="store_true", help="download the SEC ticker map again")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    rows = pd.read_csv(args.opp, usecols=ROW_COLS, low_memory=False)
    events = build_events(rows)
    log.info("%d rows -> %d events (%d owners, %d issuers)", len(rows), len(events), events["owner_cik"].nunique(),
             events["issuer_cik"].nunique())

    cands = stock.candidate_symbols(events, sources.sec_company_tickers(refresh=args.refresh_sec))
    etfs = [stock.MARKET_ETF] + sorted(set(stock.SECTOR_ETF.values()))
    sources.yahoo_prices(list(cands["symbol"].unique()) + etfs)
    cal = stock.market_calendar()
    events["anchor"] = stock.anchor_dates(events["trans_date"], cal)

    choice = stock.resolve_symbols(events, cands, cal)
    log.info("symbol status: %s", choice["symbol_status"].value_counts().to_dict())
    matched_symbols = sorted(choice["symbol"].dropna().unique())
    info = sources.yahoo_info(matched_symbols)
    sectors = {s: (info.get(s) or {}).get("sector") for s in matched_symbols}

    parts = [
        stock.price_features(events, choice, sectors, cal),
        macro.macro_features(events),
        insider.activity_features(events, insider.purchase_days(args.transactions,
                                                                args.out_dir / "cache" / "purchase_days.parquet")),
        insider.role_features(events),
    ]
    feats = pd.concat(parts, axis=1)
    feats["log_value"] = np.log1p(events.set_index("event_id")["value_usd"])

    df = (events.merge(choice, on="event_id").set_index("event_id")
          .join(feats).reset_index())
    df = df[META + FEATURES + EXTRA_FEATURES]
    for c in ("sector", "role_primary", "insider_type"):
        df[c] = df[c].astype("category")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "cfo_opp_features.parquet", index=False)
    df.to_csv(args.out_dir / "cfo_opp_features.csv", index=False, date_format="%Y-%m-%d")
    report = coverage_report(df)
    (args.out_dir / "feature_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %d events x %d features to %s", len(df), len(FEATURES) + len(EXTRA_FEATURES), args.out_dir)
    print(json.dumps({k: report[k] for k in ("events", "symbol_status", "symbol_matched_share")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
