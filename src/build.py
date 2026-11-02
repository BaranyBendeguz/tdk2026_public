# Parse, merge and clean the SEC Insider Transactions Data Sets.

from __future__ import annotations

import argparse
import glob
import gzip
import shutil
import json
import logging
import os
import re
import sys
import time
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from titles import classify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PARQUET = ROOT / "data" / "parquet"
OUT_TRANS = ROOT / "data" / "insider_transactions.csv"
OUT_HOLD = ROOT / "data" / "insider_holdings.csv"
OUT_REJECTS = ROOT / "data" / "rejects.csv"
BUILD_STATS = ROOT / "data" / "build_stats.json"

TABLES = [
    "SUBMISSION", "REPORTINGOWNER", "NONDERIV_TRANS", "NONDERIV_HOLDING",
    "DERIV_TRANS", "DERIV_HOLDING", "FOOTNOTES", "OWNER_SIGNATURE",
]
# Known header variants -> canonical name (see docs/data_cleaning.md, section 1).
COLUMN_ALIASES = {
    "EXCERCISE_DATE": "EXERCISE_DATE",
    "EXCERCISE_DATE_FN": "EXERCISE_DATE_FN",
    "FORM3_HOLDING_REPORTED": "FORM3_HOLDINGS_REPORTED",
}
OPTIONAL_COLUMNS = {"SUBMISSION": ["AFF10B5ONE"]}

TRANS_COLUMNS = [
    "accession_number", "document_type", "filing_date", "period_of_report", "issuer_cik", "issuer_name", "issuer_ticker",
    "owner_cik", "owner_name", "is_director", "is_officer", "is_ten_pct_owner", "is_other", "officer_title",
    "officer_title_clean", "role_rank",
    "table", "security_title", "trans_date", "deemed_exec_date", "trans_form_type", "trans_code", "trans_acq_disp",
    "equity_swap", "trans_timeliness",
    "trans_shares", "trans_price", "trans_value_usd", "shares_owned_after", "direct_indirect", "nature_of_ownership",
    "conv_exercise_price", "exercise_date", "expiration_date", "underlying_title", "underlying_shares",
    "aff_10b5_1_filing", "is_10b5_1_trans", "footnote_ids", "footnote_text",
    "filing_lag_days", "is_late", "is_amendment", "amended_accession", "is_superseded",
    "source_quarter",
    # extra columns, appended after the required ones
    "aff_10b5_1_available", "n_reporting_owners", "trans_date_suspect", "is_narrow_key_dup", "date_of_orig_sub",
    "trans_sk",
]
HOLD_COLUMNS = [
    "accession_number", "document_type", "filing_date", "period_of_report", "issuer_cik", "issuer_name", "issuer_ticker",
    "owner_cik", "owner_name", "is_director", "is_officer", "is_ten_pct_owner", "is_other", "officer_title",
    "officer_title_clean", "role_rank",
    "table", "security_title", "trans_form_type", "shares_owned_after", "value_owned_after", "direct_indirect",
    "nature_of_ownership", "conv_exercise_price", "exercise_date", "expiration_date", "underlying_title",
    "underlying_shares", "footnote_ids", "footnote_text",
    "is_amendment", "amended_accession", "is_superseded", "source_quarter",
    "n_reporting_owners", "date_of_orig_sub", "holding_sk",
]
REJECT_COLUMNS = ["source_quarter", "source_table", "accession_number", "row_key", "reject_reason", "raw_record"]
KEY_COLUMNS = [
    "accession_number", "owner_cik", "issuer_cik", "document_type", "filing_date", "period_of_report",
    "date_of_orig_sub", "source_quarter",
]
# The key from the specification. On the real data it also matches distinct lines of one filing (same sale
# reported for two trusts, equal gifts of two share classes), so it only sets is_narrow_key_dup.
NARROW_KEY = [
    "accession_number", "owner_cik", "table", "trans_date", "trans_code", "trans_shares", "trans_price",
    "shares_owned_after",
]
# Rows equal on every content column (everything except the surrogate key) are exact duplicates and dropped.
DEDUP_KEY = NARROW_KEY + [
    "security_title", "direct_indirect", "nature_of_ownership", "deemed_exec_date", "trans_acq_disp",
    "trans_form_type", "equity_swap", "trans_timeliness", "conv_exercise_price", "exercise_date", "expiration_date",
    "underlying_title", "underlying_shares", "footnote_ids",
]
CSV_SIZE_FACTOR = 1.4  # free disk must exceed estimated plain CSV size by this factor for --csv-compression auto

# Footnote references: F1, F12, ... A cell may hold a comma separated list.
FN_ID_RE = re.compile(r"F\d+")
FN_CELL_RE = re.compile(r"^\s*F\d+(\s*[,;]\s*F\d+)*\s*$")
# 10b5-1 with optional separators (hyphen, unicode dashes, space, underscore) between the parts:
# 10b5-1, 10b5 1, 10b-5-1, 10 b 5-1, 10b51, 10b5_1. Digit guards stop 10b5-10 or 210b5-1.
TENB51_RE = re.compile(r"(?<![0-9])10[\s_\-‐-―]*b[\s_\-‐-―]*5[\s_\-‐-―]*1(?![0-9])", re.IGNORECASE)
TRUE_VALUES = {"1", "true", "t", "y", "yes"}
FALSE_VALUES = {"0", "false", "f", "n", "no", ""}
DATE_MIN, DATE_MAX = pd.Timestamp("1900-01-01"), pd.Timestamp("2200-12-31")
TRANS_DATE_PLAUSIBLE_MIN = pd.Timestamp("1990-01-01")

log = logging.getLogger("build")


# ----------------------------------------------------------------------------- reading

def read_tsv(z: zipfile.ZipFile, table: str, quarter: str) -> tuple[pd.DataFrame, list[dict]]:
    """Read one table as strings. Lines with the wrong field count become rejects."""
    name = next((n for n in z.namelist() if n.upper() == f"{table}.TSV"), None)
    if name is None:
        raise FileNotFoundError(f"{quarter}: {table}.tsv missing from zip")
    text = z.read(name).decode("utf-8", errors="replace").replace("\r\n", "\n")
    lines = text.split("\n")
    header = [COLUMN_ALIASES.get(h.strip().upper(), h.strip().upper()) for h in lines[0].split("\t")]
    rows, bad = [], []
    for i, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != len(header):
            bad.append({
                "source_quarter": quarter, "source_table": table, "accession_number": fields[0],
                "row_key": f"line {i}", "reject_reason": f"field_count_{len(fields)}_expected_{len(header)}",
                "raw_record": line[:5000],
            })
            continue
        rows.append(fields)
    df = pd.DataFrame(rows, columns=header, dtype=object)
    for col in OPTIONAL_COLUMNS.get(table, []):
        if col not in df.columns:
            df[col] = None
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df, bad


# ----------------------------------------------------------------------------- cleaning helpers

def clean_text(s: pd.Series) -> pd.Series:
    """NFKC unicode normalization, whitespace collapsed and trimmed."""
    return s.fillna("").astype(str).str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.strip()


def parse_cik(s: pd.Series) -> pd.Series:
    """CIK string -> nullable Int64 (no leading zeros). Non-numeric -> <NA>."""
    digits = s.fillna("").str.strip()
    ok = digits.str.fullmatch(r"\d{1,10}")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out[ok] = digits[ok].astype("int64")
    return out


def parse_date(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ISO 'YYYY-MM-DD' or '', footnote ids found in the cell, unparseable-mask)."""
    raw = s.fillna("").str.strip()
    dt = pd.to_datetime(raw, format="%d-%b-%Y", errors="coerce")
    rest = dt.isna() & (raw != "") & ~raw.str.fullmatch(FN_CELL_RE.pattern)
    if rest.any():  # rare alternative formats such as 2006-01-31 or 01/31/2006
        dt[rest] = pd.to_datetime(raw[rest], format="mixed", errors="coerce")
    dt = dt.where((dt >= DATE_MIN) & (dt <= DATE_MAX))
    iso = dt.dt.strftime("%Y-%m-%d").fillna("")
    fn_ids = raw.where(raw.str.fullmatch(FN_CELL_RE.pattern), "")
    bad = (iso == "") & (raw != "") & (fn_ids == "")
    return iso, fn_ids, bad


def parse_num(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (float with NaN, footnote ids found in the cell). Strips , $ ( ) and spaces."""
    raw = s.fillna("").str.strip()
    fn_ids = raw.where(raw.str.fullmatch(FN_CELL_RE.pattern), "")
    cleaned = raw.str.replace(r"[,$()\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64"), fn_ids


def parse_bool(s: pd.Series) -> pd.Series:
    low = s.fillna("").str.strip().str.lower()
    return low.isin(TRUE_VALUES).astype("int8")


def fn_sort_key(fid: str) -> tuple[int, str]:
    m = re.search(r"\d+", fid)
    return (int(m.group()) if m else 1 << 30, fid)


def attach_footnotes(df: pd.DataFrame, fn_sources: list[pd.Series], footnotes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """footnote_ids, footnote_text, is_10b5_1_trans from the footnotes referenced by each row only.

    Also returns the number of references whose id is missing from FOOTNOTES."""
    parts = []
    for src in fn_sources:
        nz = src[src.fillna("") != ""]
        if len(nz):
            parts.append(pd.DataFrame({"rid": nz.index, "fid": nz.values}))
    out = pd.DataFrame(index=df.index)
    out["footnote_ids"], out["footnote_text"], out["is_10b5_1_trans"] = "", "", np.int8(0)
    if not parts:
        return out, 0
    long = pd.concat(parts, ignore_index=True)
    long["fid"] = long["fid"].str.findall(FN_ID_RE)
    long = long.explode("fid").dropna(subset=["fid"]).drop_duplicates(["rid", "fid"])
    long["accession_number"] = df.loc[long["rid"], "accession_number"].values
    long = long.merge(footnotes, on=["accession_number", "fid"], how="left")
    long["sk"] = long["fid"].map(fn_sort_key)
    long = long.sort_values(["rid", "sk"])
    agg = long.groupby("rid", sort=False).agg(
        footnote_ids=("fid", "|".join),
        footnote_text=("text", lambda t: " || ".join(x for x in t if isinstance(x, str) and x)),
        is_10b5_1_trans=("is10b5", "max"),
    )
    out.loc[agg.index, "footnote_ids"] = agg["footnote_ids"]
    out.loc[agg.index, "footnote_text"] = agg["footnote_text"]
    out.loc[agg.index, "is_10b5_1_trans"] = agg["is_10b5_1_trans"].fillna(0).astype("int8")
    return out, int(long["text"].isna().sum())


def raw_json(df: pd.DataFrame) -> pd.Series:
    return pd.Series([json.dumps(r, ensure_ascii=False, default=str) for r in df.to_dict("records")], index=df.index, dtype=object)


def make_rejects(raw: pd.DataFrame, mask: pd.Series, quarter: str, table: str, key_col: str, reason: pd.Series | str):
    sub = raw[mask]
    if sub.empty:
        return pd.DataFrame(columns=REJECT_COLUMNS)
    reasons = reason[mask] if isinstance(reason, pd.Series) else reason
    return pd.DataFrame({
        "source_quarter": quarter, "source_table": table, "accession_number": sub["ACCESSION_NUMBER"].values,
        "row_key": sub[key_col].values if key_col in sub else "", "reject_reason": reasons,
        "raw_record": raw_json(sub).values,
    })


# ----------------------------------------------------------------------------- submissions and owners

def build_submissions(sub: pd.DataFrame, quarter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame({"accession_number": sub["ACCESSION_NUMBER"]})
    out["document_type"] = clean_text(sub["DOCUMENT_TYPE"]).str.upper()
    out["filing_date"], _, bad_fd = parse_date(sub["FILING_DATE"])
    out["period_of_report"], _, bad_pr = parse_date(sub["PERIOD_OF_REPORT"])
    out["date_of_orig_sub"], _, _ = parse_date(sub["DATE_OF_ORIG_SUB"])
    out["issuer_cik"] = parse_cik(sub["ISSUERCIK"])
    out["issuer_name"] = clean_text(sub["ISSUERNAME"])
    out["issuer_ticker"] = clean_text(sub["ISSUERTRADINGSYMBOL"]).str.upper()
    aff = sub["AFF10B5ONE"].fillna("").str.strip().str.lower()
    out["aff_10b5_1_filing"] = aff.isin(TRUE_VALUES).astype("int8")
    out["aff_10b5_1_available"] = (aff != "").astype("int8")
    out["is_amendment"] = out["document_type"].str.endswith("/A").astype("int8")

    reason = pd.Series("", index=sub.index, dtype=object)
    reason[out["accession_number"].fillna("") == ""] = "missing_accession_number"
    reason[(reason == "") & bad_fd] = "filing_date_unparseable"
    reason[(reason == "") & (out["filing_date"] == "")] = "filing_date_missing"
    reason[(reason == "") & out["issuer_cik"].isna()] = "issuer_cik_not_integer"
    bad = reason != ""
    rejects = make_rejects(sub, bad, quarter, "SUBMISSION", "ACCESSION_NUMBER", reason)
    return out[~bad].reset_index(drop=True), rejects


def build_owners(ro: pd.DataFrame, quarter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame({"accession_number": ro["ACCESSION_NUMBER"]})
    out["owner_cik"] = parse_cik(ro["RPTOWNERCIK"])
    out["owner_name"] = clean_text(ro["RPTOWNERNAME"])
    rel = ro["RPTOWNER_RELATIONSHIP"].fillna("").str.lower().str.replace(r"[\s_]", "", regex=True)
    out["is_director"] = rel.str.contains("director").astype("int8")
    out["is_officer"] = rel.str.contains("officer").astype("int8")
    out["is_ten_pct_owner"] = rel.str.contains("tenpercentowner|10%owner|tenpercent", regex=True).astype("int8")
    out["is_other"] = rel.str.contains("other").astype("int8")
    out["officer_title"] = clean_text(ro["RPTOWNER_TITLE"])
    combos = out[["officer_title", "is_officer", "is_director", "is_ten_pct_owner"]].drop_duplicates()
    combos[["officer_title_clean", "role_rank"]] = [
        classify(t, o, d, p) for t, o, d, p in combos.itertuples(index=False)
    ]
    out = out.merge(combos, on=["officer_title", "is_officer", "is_director", "is_ten_pct_owner"], how="left")
    out["role_rank"] = out["role_rank"].astype("int8")

    reason = pd.Series("", index=ro.index, dtype=object)
    reason[out["owner_cik"].isna().values] = "owner_cik_not_integer"
    bad = reason != ""
    rejects = make_rejects(ro, bad, quarter, "REPORTINGOWNER", "RPTOWNERCIK", reason)
    good = out[~bad.values].drop_duplicates(["accession_number", "owner_cik"])
    good = good.merge(good.groupby("accession_number").size().rename("n_reporting_owners").astype("int16"),
                      on="accession_number")
    return good.reset_index(drop=True), rejects


def build_footnotes(fn: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "accession_number": fn["ACCESSION_NUMBER"],
        "fid": fn["FOOTNOTE_ID"].str.strip(),
        "text": clean_text(fn["FOOTNOTE_TXT"]),
    }).drop_duplicates(["accession_number", "fid"])
    out["is10b5"] = out["text"].str.contains(TENB51_RE).astype("int8")
    return out


# ----------------------------------------------------------------------------- transactions and holdings

def holidays() -> np.ndarray:
    return USFederalHolidayCalendar().holidays("1950-01-01", "2200-12-31").values.astype("datetime64[D]")


HOLIDAYS = holidays()


def build_rows(raw: pd.DataFrame, kind: str, table: str, quarter: str, subs: pd.DataFrame, owners: pd.DataFrame,
               footnotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """kind is 'trans' or 'hold'; table is 'nonderiv' or 'deriv'."""
    src_table = f"{table.upper()}_{'TRANS' if kind == 'trans' else 'HOLDING'}"
    sk_col = f"{src_table}_SK"
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame({"accession_number": raw["ACCESSION_NUMBER"]})
    df["table"] = table
    df["security_title"] = clean_text(raw["SECURITY_TITLE"])
    df["trans_form_type"] = raw["TRANS_FORM_TYPE"].fillna("")
    df["shares_owned_after"], fn_soa = parse_num(raw["SHRS_OWND_FOLWNG_TRANS"])
    df["direct_indirect"] = raw["DIRECT_INDIRECT_OWNERSHIP"].fillna("").str.upper()
    df["nature_of_ownership"] = clean_text(raw["NATURE_OF_OWNERSHIP"])
    extra_fn = [fn_soa]
    reason = pd.Series("", index=raw.index, dtype=object)

    if table == "deriv":
        df["conv_exercise_price"], f1 = parse_num(raw["CONV_EXERCISE_PRICE"])
        df["exercise_date"], f2, _ = parse_date(raw["EXERCISE_DATE"])
        df["expiration_date"], f3, _ = parse_date(raw["EXPIRATION_DATE"])
        df["underlying_title"] = clean_text(raw["UNDLYNG_SEC_TITLE"])
        df["underlying_shares"], f4 = parse_num(raw["UNDLYNG_SEC_SHARES"])
        extra_fn += [f1, f2, f3, f4]
    else:
        df["conv_exercise_price"] = np.nan
        df["exercise_date"] = df["expiration_date"] = df["underlying_title"] = ""
        df["underlying_shares"] = np.nan

    if kind == "trans":
        df["trans_date"], f5, bad_td = parse_date(raw["TRANS_DATE"])
        df["deemed_exec_date"], f6, _ = parse_date(raw["DEEMED_EXECUTION_DATE"])
        df["trans_code"] = raw["TRANS_CODE"].fillna("").str.upper()
        df["trans_acq_disp"] = raw["TRANS_ACQUIRED_DISP_CD"].fillna("").str.upper()
        df["equity_swap"] = parse_bool(raw["EQUITY_SWAP_INVOLVED"])
        df["trans_timeliness"] = raw["TRANS_TIMELINESS"].fillna("").str.upper()
        df["trans_shares"], f7 = parse_num(raw["TRANS_SHARES"])
        df["trans_price"], f8 = parse_num(raw["TRANS_PRICEPERSHARE"])
        df["trans_value_usd"] = df["trans_shares"] * df["trans_price"]
        df["trans_sk"] = raw[sk_col]
        extra_fn += [f5, f6, f7, f8]
        reason[bad_td] = "trans_date_unparseable"
        reason[(reason == "") & ~df["trans_acq_disp"].isin(["A", "D"])] = "trans_acq_disp_not_A_or_D"
    else:
        df["value_owned_after"], f9 = parse_num(raw["VALU_OWND_FOLWNG_TRANS"])
        df["holding_sk"] = raw[sk_col]
        extra_fn.append(f9)

    reason[(reason == "") & (df["accession_number"].fillna("") == "")] = "missing_accession_number"
    no_sub = ~df["accession_number"].isin(subs["accession_number"])
    reason[(reason == "") & no_sub] = "no_valid_submission"
    no_owner = ~df["accession_number"].isin(owners["accession_number"])
    reason[(reason == "") & no_owner] = "no_valid_reporting_owner"

    fn_cols = [raw[c] for c in raw.columns if c.endswith("_FN")]
    fn, n_missing_fn = attach_footnotes(df, fn_cols + extra_fn, footnotes)
    df = pd.concat([df, fn], axis=1)
    if kind == "hold":
        df = df.drop(columns="is_10b5_1_trans")

    bad = reason != ""
    rejects = make_rejects(raw, bad, quarter, src_table, sk_col, reason)
    df = df[~bad]
    df = df.merge(subs, on="accession_number", how="inner").merge(owners, on="accession_number", how="inner")
    df["source_quarter"] = quarter

    if kind == "trans":
        td = pd.to_datetime(df["trans_date"], errors="coerce")
        fd = pd.to_datetime(df["filing_date"], errors="coerce")
        df["trans_date_suspect"] = (td.notna() & ((td < TRANS_DATE_PLAUSIBLE_MIN) | (td > fd))).astype("int8")
        ok = td.notna() & fd.notna() & (df["trans_date_suspect"] == 0)
        lag = pd.Series(pd.NA, index=df.index, dtype="Int32")
        if ok.any():
            lag[ok] = np.busday_count(td[ok].values.astype("datetime64[D]"), fd[ok].values.astype("datetime64[D]"),
                                      holidays=HOLIDAYS)
        df["filing_lag_days"] = lag
        df["is_late"] = (lag > 2).astype("Int8")
    return df, rejects, n_missing_fn


def process_quarter(quarter: str) -> dict:
    t0 = time.time()
    path = RAW_DIR / f"{quarter}_form345.zip"
    z = zipfile.ZipFile(path)
    raw, rejects = {}, []
    for t in TABLES:
        raw[t], bad = read_tsv(z, t, quarter)
        if bad:
            rejects.append(pd.DataFrame(bad))
    headers = {t: list(raw[t].columns) for t in TABLES}

    subs, rj = build_submissions(raw["SUBMISSION"], quarter)
    rejects.append(rj)
    owners, rj = build_owners(raw["REPORTINGOWNER"], quarter)
    rejects.append(rj)
    footnotes = build_footnotes(raw["FOOTNOTES"])

    trans_parts, hold_parts, n_missing_fn = [], [], 0
    for kind, table, src in [("trans", "nonderiv", "NONDERIV_TRANS"), ("trans", "deriv", "DERIV_TRANS"),
                             ("hold", "nonderiv", "NONDERIV_HOLDING"), ("hold", "deriv", "DERIV_HOLDING")]:
        rows, rj, n = build_rows(raw[src], kind, table, quarter, subs, owners, footnotes)
        n_missing_fn += n
        (trans_parts if kind == "trans" else hold_parts).append(rows)
        rejects.append(rj)

    keys = subs.merge(owners[["accession_number", "owner_cik"]], on="accession_number")
    keys["source_quarter"] = quarter
    for kind, frame in [("trans", pd.concat(trans_parts, ignore_index=True)),
                        ("hold", pd.concat(hold_parts, ignore_index=True)),
                        ("rejects", pd.concat([r for r in rejects if len(r)] or [pd.DataFrame(columns=REJECT_COLUMNS)],
                                              ignore_index=True)),
                        ("keys", keys[KEY_COLUMNS])]:
        (INTERIM / kind).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(INTERIM / kind / f"{quarter}.parquet", index=False)
    stats = {
        "quarter": quarter, "seconds": round(time.time() - t0, 1), "headers": headers,
        "raw_rows": {t: int(len(raw[t])) for t in TABLES},
        "trans_rows": int(sum(len(p) for p in trans_parts)), "hold_rows": int(sum(len(p) for p in hold_parts)),
        "rejects": int(sum(len(r) for r in rejects)),
        "unmatched_footnote_refs": n_missing_fn,
    }
    return stats


# ----------------------------------------------------------------------------- amendments and supersession

def amendment_flags(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per (accession_number, owner_cik): amended_accession and is_superseded."""
    k = keys.copy()
    k["base"] = k["document_type"].str.replace(r"/A$", "", regex=True)
    k["is_amend"] = k["document_type"].str.endswith("/A")
    stats = {}

    # amended_accession: the original (non-amendment) filing of the same issuer, owner and form type whose
    # filing_date equals the amendment's DATE_OF_ORIG_SUB. Ties are broken by period_of_report; still ambiguous
    # or no candidate -> empty.
    am = k[k["is_amend"] & (k["date_of_orig_sub"] != "")]
    orig = k[~k["is_amend"]][["accession_number", "owner_cik", "issuer_cik", "base", "filing_date", "period_of_report"]]
    cand = am.merge(orig, left_on=["owner_cik", "issuer_cik", "base", "date_of_orig_sub"],
                    right_on=["owner_cik", "issuer_cik", "base", "filing_date"], suffixes=("", "_o"))
    n = cand.groupby(["accession_number", "owner_cik"])["accession_number_o"].transform("size")
    same_period = cand["period_of_report"] == cand["period_of_report_o"]
    n_same = same_period.groupby([cand["accession_number"], cand["owner_cik"]]).transform("sum")
    pick = cand[(n == 1) | (same_period & (n_same == 1))]
    amended = pick[["accession_number", "owner_cik", "accession_number_o"]].rename(
        columns={"accession_number_o": "amended_accession"})
    stats["amendments"] = int(k["is_amend"].sum())
    stats["amendments_with_date_of_orig_sub"] = int(len(am))
    stats["amendments_matched"] = int(len(amended))
    stats["amendments_ambiguous"] = int(cand.drop_duplicates(["accession_number", "owner_cik"]).shape[0] - len(amended))

    # is_superseded: a later amendment exists for the same (issuer_cik, owner_cik, period_of_report, base form type).
    # "Later" = filed on a later date, or on the same date when this row is an original, or on the same date with a
    # higher accession number when both are amendments.
    grp = ["issuer_cik", "owner_cik", "period_of_report", "base"]
    later = k[["accession_number", "filing_date", "is_amend"] + grp].merge(
        k[k["is_amend"]][["accession_number", "filing_date"] + grp], on=grp, suffixes=("", "_a"))
    later = later[later["accession_number"] != later["accession_number_a"]]
    cond = (later["filing_date_a"] > later["filing_date"]) | (
        (later["filing_date_a"] == later["filing_date"])
        & (~later["is_amend"] | (later["accession_number_a"] > later["accession_number"])))
    sup = later.loc[cond, ["accession_number", "owner_cik"]].drop_duplicates()
    sup["is_superseded"] = np.int8(1)
    stats["superseded_filing_owner_pairs"] = int(len(sup))

    flags = k[["accession_number", "owner_cik"]].drop_duplicates()
    flags = flags.merge(amended, on=["accession_number", "owner_cik"], how="left")
    flags = flags.merge(sup, on=["accession_number", "owner_cik"], how="left")
    flags["amended_accession"] = flags["amended_accession"].fillna("")
    flags["is_superseded"] = flags["is_superseded"].fillna(0).astype("int8")
    return flags, stats


# ----------------------------------------------------------------------------- finalize

def to_csv_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.reindex(columns=cols)


def estimate_csv_bytes(quarters: list[str]) -> int:
    """Plain CSV size estimate from a sample quarter's bytes per row and the interim row counts."""
    import pyarrow.parquet as pq
    total = 0
    for kind in ("trans", "hold"):
        rows = sum(pq.ParquetFile(INTERIM / kind / f"{q}.parquet").metadata.num_rows for q in quarters)
        sample = pd.read_parquet(INTERIM / kind / f"{quarters[-1]}.parquet")
        per_row = len(sample.to_csv(index=False).encode()) / max(len(sample), 1)
        total += int(rows * per_row)
    return total


def open_csv(path: Path, compression: str):
    if compression == "gzip":
        return gzip.open(path.with_name(path.name + ".gz"), "wt", encoding="utf-8", compresslevel=4, newline="")
    return open(path, "w", encoding="utf-8", newline="")


def finalize(quarters: list[str], write_csv: bool, compression: str = "auto") -> dict:
    keys = pd.concat([pd.read_parquet(INTERIM / "keys" / f"{q}.parquet") for q in quarters], ignore_index=True)
    dup_acc = keys.drop_duplicates(["accession_number", "source_quarter"])["accession_number"].duplicated(keep=False)
    stats = {"accessions_in_multiple_quarters": int(
        keys.drop_duplicates(["accession_number", "source_quarter"])[dup_acc]["accession_number"].nunique())}
    flags, s = amendment_flags(keys)
    stats.update(s)
    del keys

    for kind in ("transactions", "holdings"):
        (PARQUET / f"insider_{kind}").mkdir(parents=True, exist_ok=True)
    handles = {}
    if write_csv:
        if compression == "auto":
            need = estimate_csv_bytes(quarters)
            free = shutil.disk_usage(OUT_TRANS.parent).free
            compression = "none" if free > need * CSV_SIZE_FACTOR else "gzip"
            log.info("plain CSV estimate %.1f GB, free disk %.1f GB -> csv compression: %s", need / 1e9, free / 1e9,
                     compression)
        stats["csv_compression"] = compression
        for f in (OUT_TRANS, OUT_HOLD, OUT_REJECTS):
            f.unlink(missing_ok=True)
            f.with_name(f.name + ".gz").unlink(missing_ok=True)
        handles = {"trans": open_csv(OUT_TRANS, compression), "hold": open_csv(OUT_HOLD, compression),
                   "rejects": open_csv(OUT_REJECTS, "none")}
        first = True
    totals = {"transactions": 0, "holdings": 0, "rejects": 0, "duplicates_dropped": 0}
    per_quarter = {}
    for q in quarters:
        tr = pd.read_parquet(INTERIM / "trans" / f"{q}.parquet").merge(flags, on=["accession_number", "owner_cik"],
                                                                         how="left")
        ho = pd.read_parquet(INTERIM / "hold" / f"{q}.parquet").merge(flags, on=["accession_number", "owner_cik"],
                                                                        how="left")
        rj = pd.read_parquet(INTERIM / "rejects" / f"{q}.parquet")
        for d in (tr, ho):
            d["amended_accession"] = d["amended_accession"].fillna("")
            d["is_superseded"] = d["is_superseded"].fillna(0).astype("int8")

        tr["is_narrow_key_dup"] = tr.duplicated(NARROW_KEY, keep="first").astype("int8")
        dup = tr.duplicated(DEDUP_KEY, keep="first")
        if dup.any():
            dd = tr[dup]
            rj = pd.concat([rj, pd.DataFrame({
                "source_quarter": q, "source_table": dd["table"].map({"nonderiv": "NONDERIV_TRANS",
                                                                     "deriv": "DERIV_TRANS"}).values,
                "accession_number": dd["accession_number"].values, "row_key": dd["trans_sk"].values,
                "reject_reason": "duplicate_of_dedup_key",
                "raw_record": raw_json(dd[TRANS_COLUMNS]).values,
            })], ignore_index=True)
        tr = tr[~dup]
        # recompute on the kept rows so that filtering is_narrow_key_dup == 0 reproduces the narrow-key dedup
        tr["is_narrow_key_dup"] = tr.duplicated(NARROW_KEY, keep="first").astype("int8")
        tr = to_csv_frame(tr, TRANS_COLUMNS)
        ho = to_csv_frame(ho, HOLD_COLUMNS)
        tr.to_parquet(PARQUET / "insider_transactions" / f"{q}.parquet", index=False)
        ho.to_parquet(PARQUET / "insider_holdings" / f"{q}.parquet", index=False)
        if write_csv:
            for frame, k in ((tr, "trans"), (ho, "hold"), (rj.reindex(columns=REJECT_COLUMNS), "rejects")):
                frame.to_csv(handles[k], header=first, index=False)
            first = False
        totals["transactions"] += len(tr)
        totals["holdings"] += len(ho)
        totals["rejects"] += len(rj)
        totals["duplicates_dropped"] += int(dup.sum())
        per_quarter[q] = {"transactions": len(tr), "holdings": len(ho), "rejects": len(rj),
                          "duplicates_dropped": int(dup.sum())}
        log.info("%s finalized: %d transactions, %d holdings, %d rejects (%d duplicates)", q, len(tr), len(ho),
                 len(rj), int(dup.sum()))
    for fh in handles.values():
        fh.close()
    stats.update(totals)
    stats["per_quarter"] = per_quarter
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quarters", nargs="*", help="subset of quarters (default: every zip in data/raw)")
    ap.add_argument("--skip-parse", action="store_true", help="reuse existing interim parquet parts")
    ap.add_argument("--skip-csv", action="store_true", help="write parquet only")
    ap.add_argument("--csv-compression", choices=["auto", "none", "gzip"], default="auto",
                    help="auto: plain CSV if it fits on disk, else .csv.gz (rejects.csv is always plain)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    available = sorted(os.path.basename(p)[:6] for p in glob.glob(str(RAW_DIR / "*_form345.zip")))
    quarters = [q for q in available if not args.quarters or q in set(args.quarters)]
    if not quarters:
        log.error("no quarters to process in %s (run src/download.py first)", RAW_DIR)
        return 1
    log.info("processing %d quarters: %s .. %s", len(quarters), quarters[0], quarters[-1])

    failed, qstats = {}, {}
    if not args.skip_parse:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_quarter, q): q for q in quarters}
            for fut in as_completed(futs):
                q = futs[fut]
                try:
                    st = fut.result()
                    qstats[q] = st
                    log.info("%s parsed in %ss: %d transactions, %d holdings, %d rejects", q, st["seconds"],
                             st["trans_rows"], st["hold_rows"], st["rejects"])
                except Exception as exc:
                    log.exception("%s FAILED: %s", q, exc)
                    failed[q] = repr(exc)
    ok = [q for q in quarters if q not in failed]
    stats = finalize(ok, write_csv=not args.skip_csv, compression=args.csv_compression)
    stats["failed_quarters"] = failed
    stats["parse"] = {q: {k: v for k, v in s.items() if k != "headers"} for q, s in sorted(qstats.items())}
    if qstats:
        distinct = {}
        for q, s in sorted(qstats.items()):
            for t, h in s["headers"].items():
                distinct.setdefault(t, {}).setdefault(tuple(h), []).append(q)
        stats["header_variants"] = {t: [{"quarters": f"{v[0]}..{v[-1]}", "n": len(v), "columns": list(h)}
                                        for h, v in d.items()] for t, d in distinct.items()}
    BUILD_STATS.write_text(json.dumps(stats, indent=1, default=str))
    log.info("done: %d transactions, %d holdings, %d rejects, %d duplicates dropped; failed quarters: %s",
             stats["transactions"], stats["holdings"], stats["rejects"], stats["duplicates_dropped"], failed or "none")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
