# Split insider PURCHASES into routine and opportunistic ones.
import numpy as np
import pandas as pd

# column names in the Form 4 dataset
OWNER, ISSUER, DATE, FILED = "owner_cik", "issuer_cik", "trans_date", "filing_date"
CODE, ACQ = "trans_code", "trans_acq_disp"
PLAN_FLAG, FOOTNOTE = "aff_10b5_1_filing", "footnote_text"

# pattern rules
LOOKBACK_DAYS = 3 * 365 + 31     # only the last ~3 years of behaviour count
EPISODE_GAP_DAYS = 5             # buys this close to the previous one = same episode
MIN_GAPS = 3                     # need >= 3 gaps (4 previous episodes) to call a schedule
MAX_GAPS = 6                     # look at the most recent 6 gaps
GAP_TOL = 0.20                   # "same length" = within 20% of the median gap ...
MIN_TOL_DAYS = 3                 # ... but at least +/- 3 days
MAX_MULTIPLE = 2                 # a gap of ~2x the period (one skipped period) is on schedule
ON_SCHEDULE_SHARE = 0.75         # share of gaps that must be on schedule
ANNUAL_YEARS = 3                 # same month in each of the previous 3 years

OUT_COLS = ["known_date", "prev_episodes", "median_gap_days", "on_schedule_share",
            "same_month_prev_years", "planned_10b5_1", "periodic_gaps", "annual_pattern",
            "is_routine", "is_opportunistic"]


def purchase_mask(df: pd.DataFrame) -> pd.Series:
    """Open-market / private purchases (code P, acquired), from every filing (originals,
    amendments and re-reports alike)."""
    m = (df[CODE].astype(str).str.strip().str.upper() == "P") & \
        (df[ACQ].astype(str).str.strip().str.upper() == "A")
    return m & df[OWNER].notna() & df[ISSUER].notna() & df[DATE].notna() & df[FILED].notna()


def planned_10b5_1(df: pd.DataFrame) -> pd.Series:
    """10b5-1 plan trade, judged from the row's own filing."""
    flag = pd.Series(False, index=df.index)
    if PLAN_FLAG in df.columns:
        flag |= pd.to_numeric(df[PLAN_FLAG], errors="coerce").fillna(0) == 1
    if FOOTNOTE in df.columns:
        text = df[FOOTNOTE].fillna("").astype(str)
        mentions = text.str.contains(r"10b5[\s\-–]*1", case=False, regex=True)
        negated = text.str.contains(r"\bnot\b[^.]{0,60}10b5", case=False, regex=True)
        flag |= mentions & ~negated
    return flag


def _day_patterns(day: np.ndarray, known: np.ndarray, ym: np.ndarray,
                  starts: np.ndarray, ends: np.ndarray):
    """day: int day numbers of buy days, sorted within each (owner, issuer) group; known: the
    first filing date of each buy day. Returns per-day arrays computed from the group's earlier
    buy days that were public on that day's known date (the day itself included)."""
    n = len(day)
    prev = np.zeros(n, np.int32)
    med = np.full(n, np.nan)
    share = np.full(n, np.nan)
    years = np.zeros(n, np.int8)
    for a, b in zip(starts, ends):
        t, f, m = day[a:b], known[a:b], ym[a:b]
        for k in range(b - a):
            vis = f[:k + 1] <= f[k]
            td, tm = t[:k + 1][vis], m[:k + 1][vis]
            new_ep = np.r_[True, np.diff(td) > EPISODE_GAP_DAYS]
            st, sm = td[new_ep], tm[new_ep]             # episode starts; the last one is the current
            cur, cur_m = st[-1], sm[-1]
            past = st[:-1][st[:-1] >= cur - LOOKBACK_DAYS]
            prev[a + k] = len(past)
            if len(past) >= MIN_GAPS + 1:
                gaps = np.diff(past[-(MAX_GAPS + 1):])
                g = np.median(gaps)
                med[a + k] = g
                mult = np.clip(np.round(gaps / g), 1, MAX_MULTIPLE)
                share[a + k] = np.mean(np.abs(gaps - mult * g) <= max(GAP_TOL * g, MIN_TOL_DAYS))
            seen = set(sm[:-1].tolist())
            y = 0
            while y < ANNUAL_YEARS and (cur_m - 12 * (y + 1)) in seen:
                y += 1
            years[a + k] = y
    return prev, med, share, years


def add_buy_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with the OUT_COLS columns. Non-purchase rows get NaN / False."""
    buys = purchase_mask(df)
    b = pd.DataFrame({"owner": df.loc[buys, OWNER].astype(str),
                      "issuer": df.loc[buys, ISSUER].astype(str),
                      "day": pd.to_datetime(df.loc[buys, DATE], errors="coerce"),
                      "filed": pd.to_datetime(df.loc[buys, FILED], errors="coerce"),
                      "planned": planned_10b5_1(df.loc[buys])})
    b = b.dropna(subset=["day", "filed"])
    b["day"] = b["day"].dt.normalize()

    # one row per insider / company / buy day, dated and flagged by the first filing that reported it
    key3 = ["owner", "issuer", "day"]
    first = b[b["filed"] == b.groupby(key3)["filed"].transform("min")]
    days = first.groupby(key3, as_index=False, sort=True).agg(known_date=("filed", "first"),
                                                              planned_10b5_1=("planned", "any"))
    dnum = days["day"].to_numpy("datetime64[D]").astype(np.int64)
    knum = days["known_date"].to_numpy("datetime64[D]").astype(np.int64)
    ym = (days["day"].dt.year * 12 + days["day"].dt.month - 1).to_numpy()
    key = (days["owner"] + "|" + days["issuer"]).to_numpy()
    g_starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    g_ends = np.r_[g_starts[1:], len(days)]
    prev, med, share, years = _day_patterns(dnum, knum, ym, g_starts, g_ends)
    days["prev_episodes"], days["median_gap_days"] = prev, med
    days["on_schedule_share"], days["same_month_prev_years"] = share, years

    rows = b[key3].rename_axis("_row").reset_index() \
                  .merge(days, on=key3, how="left").set_index("_row")
    out = df.copy()
    for c in ["known_date", "prev_episodes", "median_gap_days", "on_schedule_share",
              "same_month_prev_years", "planned_10b5_1"]:
        out[c] = rows[c].reindex(df.index)
    out["planned_10b5_1"] = out["planned_10b5_1"].fillna(False).astype(bool)
    out["periodic_gaps"] = (out["prev_episodes"] >= MIN_GAPS + 1) & \
                           (out["on_schedule_share"] >= ON_SCHEDULE_SHARE)
    out["annual_pattern"] = out["same_month_prev_years"] >= ANNUAL_YEARS
    is_buy = out["prev_episodes"].notna()
    out["is_routine"] = is_buy & (out["planned_10b5_1"] | out["periodic_gaps"]
                                  | out["annual_pattern"])
    out["is_opportunistic"] = is_buy & ~out["is_routine"]
    return out


def is_opportunistic(df: pd.DataFrame, min_prev_episodes: int = 0) -> pd.Series:
    """True for purchases with no periodic pattern in the insider's past buys.
    min_prev_episodes > 0 also requires that much buying history (with 0, an insider's
    first-ever buy counts as opportunistic, since it has no pattern)."""
    p = add_buy_patterns(df)
    return p["is_opportunistic"] & (p["prev_episodes"].fillna(0) >= min_prev_episodes)


def pattern_report(p: pd.DataFrame) -> dict:
    """Counts by reason, for a frame returned by add_buy_patterns. Prints and returns them."""
    buys = p["prev_episodes"].notna()
    n = int(buys.sum())
    counts = {"purchases": n}
    print(f"purchases: {n:,}")
    for c in ["planned_10b5_1", "annual_pattern", "periodic_gaps", "is_routine", "is_opportunistic"]:
        counts[c] = int(p.loc[buys, c].sum())
        print(f"  {c:<18} {counts[c]:>10,}  ({p.loc[buys, c].mean():.1%})")
    counts["first_buy_share"] = round(float((p.loc[buys, "prev_episodes"] == 0).mean()), 4)
    print(f"  first buy (no history): {counts['first_buy_share']:.1%}")
    return counts
