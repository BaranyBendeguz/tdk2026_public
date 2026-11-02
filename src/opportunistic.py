# Select the opportunistic purchases from the CFO insider transactions.

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "opportunistic_buys"
DEFAULT_INPUT = OUT_DIR / "cfo_insider_transactions.csv"
DATA_START = pd.Timestamp("2006-01-01")  # first quarter of the SEC data sets

LOOKBACK_DAYS = 3 * 365
BURST_GAP_DAYS = 4  # covers a weekend plus a holiday; weekly buying (gap 6-8) stays separate
MIN_REGULAR_GAPS = 3  # 4 buying episodes in a row
PERIOD_TOL = 0.15
PERIODS = {
    "weekly": 7.0,
    "biweekly": 14.0,
    "monthly": 365.25 / 12,
    "quarterly": 365.25 / 4,
    "semiannual": 365.25 / 2,
}
CALENDAR_YEARS = 3

EVENT_KEY = ["owner_cik", "issuer_cik", "trans_date"]
USECOLS = [
    "accession_number", "document_type", "filing_date", "issuer_cik", "issuer_name", "issuer_ticker", "owner_cik",
    "owner_name", "is_director", "is_ten_pct_owner", "officer_title", "table", "security_title", "trans_date",
    "trans_form_type", "trans_code", "trans_acq_disp", "trans_shares", "trans_price", "shares_owned_after",
    "direct_indirect", "nature_of_ownership", "aff_10b5_1_filing", "is_10b5_1_trans", "footnote_text",
    "filing_lag_days", "is_amendment", "trans_date_suspect",
]

# Security titles: these are never the common stock.
NON_EQUITY_RE = re.compile(
    r"warrant|\boptions?\b|\bnotes?\b|debenture|\bbonds?\b|phantom|deferred|\bdebt\b|subordinated", re.I)
# These are not the common stock unless the title also says common ("Common Stock and associated Preferred Stock
# Purchase Rights" is common stock).
PREFERRED_RE = re.compile(r"prefer|pfd|\brights?\b", re.I)
COMMON_RE = re.compile(r"common|ordinary", re.I)

# Plan purchases. Matched per footnote, because the pipeline joins the footnotes of every field of a row: a note on
# shares_owned_after such as "Includes 764 shares acquired under the ESPP" describes the holding, not this purchase.
PLAN_RE = re.compile(
    r"employee (?:stock|share)s? purchase|\bespp\b|stock purchase plan|direct stock purchase|dividend reinvest"
    r"|reinvest\w* (?:of )?(?:cash )?dividends?|dividends? (?:were |was )?reinvested|\bdrip\b|401\s*\(?k\)?"
    r"|payroll deduction|deferred compensation|savings plan|thrift plan|\besop\b|employee stock ownership", re.I)
NOT_PLAN_NOTE_RE = re.compile(
    r"\binclud|\breflects?\b|beneficially own|following (?:the|this|these)|total (?:holdings|shares|number)"
    r"|held (?:in|by|through)|holdings|discretionary", re.I)

log = logging.getLogger("opportunistic")


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet" or path.is_dir():
        return pd.read_parquet(path, columns=USECOLS)
    return pd.read_csv(path, usecols=USECOLS, dtype={"trans_form_type": str}, low_memory=False)


def common_equity_mask(title: pd.Series) -> pd.Series:
    t = title.fillna("")
    non_equity = t.str.contains(NON_EQUITY_RE)
    preferred = t.str.contains(PREFERRED_RE) & ~t.str.contains(COMMON_RE)
    return ~(non_equity | preferred)


def plan_purchase_mask(df: pd.DataFrame) -> pd.Series:
    notes = df["footnote_text"].fillna("").str.split(" || ", regex=False).explode()
    by_note = (notes.str.contains(PLAN_RE) & ~notes.str.contains(NOT_PLAN_NOTE_RE)).groupby(level=0).any()
    by_nature = df["nature_of_ownership"].fillna("").str.contains(PLAN_RE)
    return by_note.reindex(df.index, fill_value=False) | by_nature


def select_purchases(df: pd.DataFrame, stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (universe, plan): the open-market purchase rows and the plan purchase rows that were dropped."""
    steps = stats["row_filters"] = {"input_rows": len(df)}
    filters = [
        ("trans_code_P", df["trans_code"].eq("P")),
        ("acquired", df["trans_acq_disp"].eq("A")),
        ("table_I_nonderiv", df["table"].eq("nonderiv")),
        ("trans_date_plausible", df["trans_date_suspect"].eq(0)),
        ("shares_positive", df["trans_shares"].gt(0)),
        ("common_equity", common_equity_mask(df["security_title"])),
    ]
    keep = pd.Series(True, index=df.index)
    for name, m in filters:
        keep &= m
        steps[name] = int(keep.sum())
    df = df[keep].copy()
    plan = plan_purchase_mask(df)
    steps["not_plan_purchase"] = int((~plan).sum())
    for c in ("trans_date", "filing_date"):
        df[c] = pd.to_datetime(df[c])
    return df[~plan].copy(), df[plan].copy()


def build_events(rows: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """One row per (owner, issuer, trans_date), from the rows of the first filing date that reported it."""
    first = rows.groupby(EVENT_KEY)["filing_date"].transform("min")
    stats["rows_from_later_reports_dropped"] = int((rows["filing_date"] != first).sum())
    rows = rows[rows["filing_date"] == first]
    # the same line filed twice on one day (separate accessions) counts once
    n = len(rows)
    rows = rows.drop_duplicates(EVENT_KEY + ["security_title", "trans_shares", "trans_price", "shares_owned_after",
                                             "direct_indirect", "nature_of_ownership"])
    stats["same_day_duplicate_rows_dropped"] = n - len(rows)

    priced = rows["trans_price"] > 0
    rows = rows.assign(
        priced_shares=rows["trans_shares"].where(priced),
        priced_value=(rows["trans_shares"] * rows["trans_price"]).where(priced),
        priced_price=rows["trans_price"].where(priced),
        direct_after=rows["shares_owned_after"].where(rows["direct_indirect"].eq("D")),
        indirect=rows["direct_indirect"].eq("I"),
        is_10b5_1=(rows["is_10b5_1_trans"] == 1) | (rows["aff_10b5_1_filing"] == 1),
    )
    g = rows.groupby(EVENT_KEY, sort=True)
    ev = g.agg(
        owner_name=("owner_name", "first"),
        issuer_name=("issuer_name", "first"),
        issuer_ticker=("issuer_ticker", "first"),
        signal_date=("filing_date", "first"),
        filing_lag_days=("filing_lag_days", "min"),
        n_filings=("accession_number", "nunique"),
        first_report_amendment=("is_amendment", "min"),
        trans_form_type=("trans_form_type", "first"),
        officer_title=("officer_title", "first"),
        is_director=("is_director", "max"),
        is_ten_pct_owner=("is_ten_pct_owner", "max"),
        n_fills=("trans_shares", "size"),
        shares=("trans_shares", "sum"),
        priced_shares=("priced_shares", "sum"),
        value_usd=("priced_value", lambda s: s.sum(min_count=1)),
        price_min=("priced_price", "min"),
        price_max=("priced_price", "max"),
        shares_owned_after_direct=("direct_after", "max"),
        has_indirect=("indirect", "max"),
        is_10b5_1=("is_10b5_1", "max"),
    )
    ev["accession_numbers"] = g["accession_number"].agg(lambda s: "|".join(sorted(set(s))))
    ev["vwap"] = ev["value_usd"] / ev["priced_shares"].where(ev["priced_shares"] > 0)
    ev["has_price"] = ev["vwap"].notna()
    for c in ("has_indirect", "is_10b5_1", "has_price"):
        ev[c] = ev[c].astype(int)
    return ev.drop(columns="priced_shares").reset_index()


def buy_day_history(rows: pd.DataFrame) -> pd.DataFrame:
    """Distinct buy-days per owner (all issuers) and the date each one became public (first filing)."""
    return (rows.groupby(["owner_cik", "trans_date"], sort=True)["filing_date"].min()
            .rename("known_date").reset_index())


def longest_run(ok: np.ndarray) -> int:
    best = cur = 0
    for v in ok:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def classify(events: pd.DataFrame, hist: pd.DataFrame, lookback_days: int, burst_gap_days: int,
             min_regular_gaps: int, period_tol: float) -> pd.DataFrame:
    """Routine flags for every event, using only the owner's buy-days that were public on its signal_date."""
    day = np.timedelta64(1, "D")
    by_owner = {}
    for owner, h in hist.groupby("owner_cik", sort=False):
        d = h["trans_date"].to_numpy()
        by_owner[owner] = (
            (d - np.datetime64("1970-01-01")) // day,
            (h["known_date"].to_numpy() - np.datetime64("1970-01-01")) // day,
            h["trans_date"].dt.year.to_numpy(),
            h["trans_date"].dt.month.to_numpy(),
        )

    t_all = (events["trans_date"].to_numpy() - np.datetime64("1970-01-01")) // day
    f_all = (events["signal_date"].to_numpy() - np.datetime64("1970-01-01")) // day
    y_all = events["trans_date"].dt.year.to_numpy()
    out = {k: [] for k in ("hist_n_buy_days", "hist_days_since_prev", "hist_n_episodes", "hist_years_with_buys",
                           "episode_start", "day_in_episode", "routine_cadence", "routine_period",
                           "routine_calendar", "routine_calendar_month")}

    for owner, t, f, y in zip(events["owner_cik"].to_numpy(), t_all, f_all, y_all):
        d, known, yr, mo = by_owner[owner]
        sel = (d < t) & (known <= f)
        prior = d[sel]

        # calendar rule: a month with buys in each of the CALENDAR_YEARS preceding calendar years
        months = [set(mo[sel & (yr == y - k)]) for k in range(1, CALENDAR_YEARS + 1)]
        common = set.intersection(*months)

        # cadence rule on episode starts, the current episode included
        seq = np.append(prior[prior >= t - lookback_days], t)
        is_start = np.r_[True, np.diff(seq) > burst_gap_days]
        starts = seq[is_start]
        gaps = np.diff(starts)
        period = ""
        for name, p in PERIODS.items():
            ok = (gaps >= p * (1 - period_tol)) & (gaps <= p * (1 + period_tol))
            if longest_run(ok) >= min_regular_gaps:
                period = name
                break

        out["hist_n_buy_days"].append(len(prior))
        out["hist_days_since_prev"].append(t - prior[-1] if len(prior) else np.nan)
        out["hist_n_episodes"].append(len(starts) - 1)
        out["hist_years_with_buys"].append(sum(bool(m) for m in months))
        out["episode_start"].append(starts[-1])
        out["day_in_episode"].append(int(len(seq) - np.flatnonzero(is_start)[-1]))
        out["routine_cadence"].append(int(bool(period)))
        out["routine_period"].append(period)
        out["routine_calendar"].append(int(bool(common)))
        out["routine_calendar_month"].append(min(common) if common else np.nan)

    res = pd.DataFrame(out, index=events.index)
    res["episode_start"] = pd.to_datetime(res["episode_start"], unit="D")
    return res


def summarize(ev: pd.DataFrame, stats: dict) -> None:
    stats["events"] = len(ev)
    stats["event_flags"] = {
        "routine_cadence": int(ev["routine_cadence"].sum()),
        "routine_calendar": int(ev["routine_calendar"].sum()),
        "routine_any": int(ev["routine"].sum()),
        "is_10b5_1": int(ev["is_10b5_1"].sum()),
        "no_price": int((ev["has_price"] == 0).sum()),
        "history_censored": int(ev["history_censored"].sum()),
        "is_opportunistic": int(ev["is_opportunistic"].sum()),
    }
    stats["routine_period"] = ev.loc[ev["routine_cadence"] == 1, "routine_period"].value_counts().to_dict()
    opp = ev[ev["is_opportunistic"] == 1]
    stats["opportunistic"] = {
        "events": len(opp),
        "owners": int(opp["owner_cik"].nunique()),
        "issuers": int(opp["issuer_cik"].nunique()),
        "episodes": int(opp.groupby(["owner_cik", "episode_start"]).ngroups),
        "first_buy_of_owner": int((opp["hist_n_buy_days"] == 0).sum()),
        "per_signal_year": {int(k): int(v) for k, v in opp["signal_date"].dt.year.value_counts().sort_index().items()},
    }
    by_year = ev.groupby(ev["signal_date"].dt.year).agg(events=("is_opportunistic", "size"),
                                                         opportunistic=("is_opportunistic", "sum"),
                                                         routine=("routine", "sum"))
    stats["per_signal_year_all"] = {int(k): {c: int(v) for c, v in r.items()} for k, r in by_year.iterrows()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CFO transactions (csv, parquet or directory)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--burst-gap-days", type=int, default=BURST_GAP_DAYS)
    ap.add_argument("--min-regular-gaps", type=int, default=MIN_REGULAR_GAPS)
    ap.add_argument("--period-tol", type=float, default=PERIOD_TOL)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    stats: dict = {"params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    stats["params"]["periods_days"] = PERIODS
    log.info("reading %s", args.input)
    raw = read_input(args.input)
    rows, plan = select_purchases(raw, stats)
    del raw
    log.info("open-market purchase rows: %d (plan purchase rows dropped: %d)", len(rows), len(plan))

    ev = build_events(rows, stats)
    hist = buy_day_history(rows)
    log.info("buy events: %d, owners: %d, owner buy-days: %d", len(ev), ev["owner_cik"].nunique(), len(hist))

    ev = ev.join(classify(ev, hist, args.lookback_days, args.burst_gap_days, args.min_regular_gaps, args.period_tol))
    ev["routine"] = ((ev["routine_cadence"] == 1) | (ev["routine_calendar"] == 1)).astype(int)
    # before 2009 the 3-year history reaches back before the first data set, so routine buyers can look new
    lookback_start = ev["trans_date"] - pd.to_timedelta(args.lookback_days, unit="D")
    ev["history_censored"] = ((lookback_start < DATA_START)
                              | (ev["trans_date"].dt.year - CALENDAR_YEARS < DATA_START.year)).astype(int)
    ev["is_opportunistic"] = ((ev["routine"] == 0) & (ev["is_10b5_1"] == 0) & (ev["has_price"] == 1)).astype(int)
    summarize(ev, stats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(args.out_dir / "cfo_buy_events.parquet", index=False)
    opp = ev[ev["is_opportunistic"] == 1].reset_index(drop=True)
    opp.to_parquet(args.out_dir / "cfo_opportunistic_buys.parquet", index=False)
    opp.to_csv(args.out_dir / "cfo_opportunistic_buys.csv", index=False, date_format="%Y-%m-%d")
    with open(args.out_dir / "opportunistic_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=int)

    print(json.dumps({k: stats[k] for k in ("row_filters", "events", "event_flags", "routine_period")}, indent=2))
    print(json.dumps({k: v for k, v in stats["opportunistic"].items() if k != "per_signal_year"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
