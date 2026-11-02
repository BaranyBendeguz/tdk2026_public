# Insider activity features (from all insiders' purchases) and role features (from the reported title).

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from titles import ROLE_RE, normalize_title  # noqa: E402

PURCHASE_COLS = ["issuer_cik", "owner_cik", "trans_date", "filing_date", "trans_code", "trans_acq_disp", "table",
                 "trans_shares", "trans_price", "trans_date_suspect"]
DAY = ["issuer_cik", "owner_cik", "trans_date"]
ACTIVITY_FEATURES = ["ins_n_insiders_30d", "ins_n_insiders_90d", "ins_log_value_30d", "ins_log_value_90d",
                     "ins_days_since_prev", "ins_n_prev_buys_365d"]

ROLE_FLAGS = ["role_ceo", "role_cfo", "role_coo", "role_csuite_other", "role_president", "role_chair",
              "role_evp_svp", "role_vp", "role_finance_other", "role_legal", "role_officer_other", "role_director",
              "role_owner_10pct", "role_founder", "role_former", "role_other_role"]
ROLE_FEATURES = ROLE_FLAGS + ["role_primary", "insider_type", "exec_seniority", "n_roles", "title_missing",
                              "insider_is_entity"]
# most senior first, for role_primary
ROLE_ORDER = ["ceo", "chair", "president", "cfo", "coo", "csuite_other", "evp_svp", "legal", "finance_other", "vp",
              "officer_other", "director", "owner_10pct", "founder", "other_role", "former"]
SENIORITY = {"ceo": 5, "president": 4, "cfo": 4, "coo": 4, "csuite_other": 3, "evp_svp": 2, "vp": 1}

_CSUITE_OTHER = re.compile(
    r"\bchief (?!executive\b|financial\b|finance\b|operating\b)[a-z]+(?: [a-z]+)? officer\b|\bchief legal\b"
    r"|\b(?:cio|cto|cao|cmo|cco|cro|chro|clo|cso|cdo|cpo|cno|ciso)\b|\bprincipal accounting\b")
_PRESIDENT = re.compile(r"\bpresident\b")
_VICE_PRES = re.compile(r"\b(?:vice president|vp|evp|svp|avp)\b")
_CHAIR = re.compile(r"\bchairman\b")
_VICE_CHAIR = re.compile(r"\bvice chairman\b")
_EVP_SVP = re.compile(r"\b(?:executive|senior|group executive) (?:vice president|vp)\b|\bevp\b|\bsvp\b|\bsevp\b")
_VP = re.compile(r"\bvice president\b|\bvp\b|\bavp\b")
_FINANCE = re.compile(r"\btreasurer\b|\bcontroller\b|\bcomptroller\b|\bfinanc\w*|\baccounting\b")
_LEGAL = re.compile(r"\bgeneral counsel\b|\bgc\b|\bchief legal\b|\blegal\b|\bcounsel\b|\bsecretary\b")
_DIRECTOR = re.compile(
    r"(?:^|\band\b|\b(?:independent|lead|outside|non executive|board|presiding)\s+)directors?\b(?!\s+of\b)"
    r"|\bboard member\b|\bmember of (?:the )?board\b")
_FOUNDER = re.compile(r"\b(?:co ?)?founder\b")
_FORMER = re.compile(r"\bformer(?:ly)?\b|\bfmr\b|\bex\b|\bretired\b|\bresigned\b|\bpast\b")
_ENTITY = re.compile(
    r"\b(?:llc|l l c|inc|incorporated|corp|corporation|company|co|lp|l p|llp|ltd|limited|trust|fund|funds|partners"
    r"|partnership|holdings?|capital|management|group|bank|bancorp|foundation|associates|investments?|ventures"
    r"|advisors|advisers|plc|gmbh|ag|sa|nv|bv)\b")

log = logging.getLogger("features.insider")


def purchase_days(transactions: Path, cache: Path) -> pd.DataFrame:
    """All insiders' open-market buy-days (issuer, owner, trans_date) with the date the first filing became public
    (known_date) and the dollar value as first reported. Cached as parquet; built from the full CSV once."""
    if cache.exists() and cache.stat().st_mtime > transactions.stat().st_mtime:
        return pd.read_parquet(cache)
    log.info("reading purchases from %s (one pass over the full file)", transactions)
    parts = []
    reader = pd.read_csv(transactions, usecols=PURCHASE_COLS, chunksize=1_000_000,
                         dtype={"trans_code": str, "trans_acq_disp": str, "table": str})
    for n, ch in enumerate(reader):
        m = (ch["trans_code"].eq("P") & ch["trans_acq_disp"].eq("A") & ch["table"].eq("nonderiv")
             & ch["trans_date_suspect"].eq(0) & ch["trans_shares"].gt(0))
        parts.append(ch.loc[m, ["issuer_cik", "owner_cik", "trans_date", "filing_date", "trans_shares",
                                "trans_price"]])
        if n % 2 == 0:
            log.info("  %d M rows read", n + 1)
    rows = pd.concat(parts, ignore_index=True)
    for c in ("trans_date", "filing_date"):
        rows[c] = pd.to_datetime(rows[c])
    first = rows.groupby(DAY)["filing_date"].transform("min")
    rows = rows[rows["filing_date"] == first]
    rows["value_usd"] = (rows["trans_shares"] * rows["trans_price"]).where(rows["trans_price"] > 0, 0.0)
    days = rows.groupby(DAY, as_index=False).agg(known_date=("filing_date", "first"), value_usd=("value_usd", "sum"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    days.to_parquet(cache, index=False)
    return days


def activity_features(events: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Indexed by event_id. Windows are calendar days up to and including trans_date."""
    epoch = np.datetime64("1970-01-01")
    one = np.timedelta64(1, "D")
    days = days[days["issuer_cik"].isin(events["issuer_cik"])].sort_values(["issuer_cik", "trans_date"])
    by_issuer = {
        i: ((g["trans_date"].to_numpy() - epoch) // one, (g["known_date"].to_numpy() - epoch) // one,
            g["owner_cik"].to_numpy(), g["value_usd"].to_numpy())
        for i, g in days.groupby("issuer_cik")
    }
    empty = (np.array([], int), np.array([], int), np.array([], int), np.array([], float))
    t_all = (events["trans_date"].to_numpy() - epoch) // one
    s_all = (events["signal_date"].to_numpy() - epoch) // one
    rows = []
    for issuer, owner, t, s in zip(events["issuer_cik"], events["owner_cik"], t_all, s_all):
        td, known, ow, val = by_issuer.get(issuer, empty)
        vis = known <= s
        r = []
        for w in (30, 90):
            m = vis & (td >= t - w) & (td <= t)
            r += [len(np.unique(ow[m])), np.log1p(val[m].sum())]
        prev = td[vis & (ow == owner) & (td < t)]
        r += [t - prev.max() if len(prev) else np.nan, int((prev >= t - 365).sum())]
        rows.append(r)
    out = pd.DataFrame(rows, index=events["event_id"],
                       columns=["ins_n_insiders_30d", "ins_log_value_30d", "ins_n_insiders_90d",
                                "ins_log_value_90d", "ins_days_since_prev", "ins_n_prev_buys_365d"])
    return out[ACTIVITY_FEATURES]


def title_flags(title: str, is_officer: int, is_director: int, is_ten_pct: int, is_other: int) -> dict:
    s = normalize_title(title)
    no_cfo = ROLE_RE["cfo"].sub(" ", s)
    no_evp = _EVP_SVP.sub(" ", s)
    f = {
        "ceo": bool(ROLE_RE["ceo"].search(s)),
        "cfo": bool(ROLE_RE["cfo"].search(s)),
        "coo": bool(ROLE_RE["coo"].search(s)),
        "csuite_other": bool(_CSUITE_OTHER.search(s)),
        "president": bool(_PRESIDENT.search(_VICE_PRES.sub(" ", s))),
        "chair": bool(_CHAIR.search(_VICE_CHAIR.sub(" ", s))),
        "evp_svp": bool(_EVP_SVP.search(s)),
        "vp": bool(_VP.search(no_evp)),
        "finance_other": bool(_FINANCE.search(no_cfo)),
        "legal": bool(_LEGAL.search(s)),
        "director": bool(is_director) or bool(_DIRECTOR.search(s)) or bool(_VICE_CHAIR.search(s)),
        "owner_10pct": bool(is_ten_pct) or "tenpctowner" in s,
        "founder": bool(_FOUNDER.search(s)),
        "former": bool(_FORMER.search(s)),
        "other_role": bool(is_other),
    }
    officer_roles = ("ceo", "cfo", "coo", "csuite_other", "president", "evp_svp", "vp", "finance_other", "legal")
    f["officer_other"] = bool(is_officer) and not any(f[r] for r in officer_roles)
    return f


def role_features(events: pd.DataFrame) -> pd.DataFrame:
    """Indexed by event_id."""
    recs = []
    for title, off, dire, ten, oth, name in events[["officer_title", "is_officer", "is_director", "is_ten_pct_owner",
                                                   "is_other", "owner_name"]].itertuples(index=False):
        title = title if isinstance(title, str) else ""
        f = title_flags(title, off, dire, ten, oth)
        r = {f"role_{k}": int(v) for k, v in f.items()}
        r["role_primary"] = next((k for k in ROLE_ORDER if f[k]), "none")
        r["insider_type"] = ("officer and director" if off and dire else "officer" if off else "director" if dire
                             else "10% owner" if ten else "other" if oth else "missing")
        r["exec_seniority"] = max([SENIORITY.get(k, 0) for k, v in f.items() if v], default=0)
        r["n_roles"] = int(sum(f.values()))
        r["title_missing"] = int(not title.strip())
        r["insider_is_entity"] = int(bool(_ENTITY.search(normalize_title(name if isinstance(name, str) else ""))))
        recs.append(r)
    return pd.DataFrame(recs, index=events["event_id"])[ROLE_FEATURES]
