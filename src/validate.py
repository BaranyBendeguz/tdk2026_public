# Validate the built insider transactions data set.

from __future__ import annotations

import argparse
import gzip
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import HOLD_COLUMNS, OUT_HOLD, OUT_TRANS, PARQUET, TRANS_COLUMNS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "validation_report.txt"
DATE_COLS = ["filing_date", "period_of_report", "trans_date", "deemed_exec_date", "exercise_date", "expiration_date"]
TRANS_KEY = ["accession_number", "owner_cik", "table", "trans_sk"]
TODAY = pd.Timestamp(date.today())


def csv_rows(path: Path) -> int | None:
    for p in (path, path.with_name(path.name + ".gz")):
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rb") as fh:
                return sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 24), b"")) - 1
    return None


def nullish(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.isna() | (s.astype(str) == "")
    return s.isna()


def run(skip_csv_count: bool) -> list[str]:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    parts = sorted((PARQUET / "insider_transactions").glob("*.parquet"))
    hparts = sorted((PARQUET / "insider_holdings").glob("*.parquet"))
    if not parts:
        print("no parquet partitions found; run src/build.py first")
        return ["no data"]

    rows_q, rows_table, nulls, codes = {}, Counter(), Counter(), Counter()
    n = n10b5 = nlate = nlate_known = nsuspect = nnarrow = nsuperseded = namend = namend_linked = 0
    dup_keys = bad_dates = bad_value = bad_ad = bad_flags = bad_iso = 0
    min_fd, max_fd, min_td, max_td = None, None, None, None
    acc_quarters: dict[str, str] = {}
    acc_multi = 0
    columns_ok = True

    for p in parts:
        q = p.stem
        df = pd.read_parquet(p)
        columns_ok &= list(df.columns) == TRANS_COLUMNS
        rows_q[q] = len(df)
        n += len(df)
        rows_table.update(df["table"].value_counts().to_dict())
        for c in df.columns:
            nulls[c] += int(nullish(df[c]).sum())
        codes.update(df["trans_code"].replace("", "(empty)").value_counts().to_dict())
        n10b5 += int(df["is_10b5_1_trans"].sum())
        nlate += int(df["is_late"].fillna(0).sum())
        nlate_known += int(df["is_late"].notna().sum())
        nsuspect += int(df["trans_date_suspect"].sum())
        nnarrow += int(df["is_narrow_key_dup"].sum())
        nsuperseded += int(df["is_superseded"].sum())
        namend += int(df["is_amendment"].sum())
        namend_linked += int((df["is_amendment"].astype(bool) & (df["amended_accession"] != "")).sum())

        dup_keys += int(df.duplicated(TRANS_KEY).sum())
        for c in DATE_COLS:
            v = df[c][df[c] != ""]
            bad_iso += int((~v.str.fullmatch(r"\d{4}-\d{2}-\d{2}")).sum())
        fd = pd.to_datetime(df["filing_date"])
        td = pd.to_datetime(df["trans_date"].replace("", None))
        bad_dates += int(((fd < "2006-01-01") | (fd > TODAY)).sum())
        ok_td = td.notna() & (df["trans_date_suspect"] == 0)
        bad_dates += int(((td[ok_td] < "1990-01-01") | (td[ok_td] > fd[ok_td])).sum())
        for c in ("deemed_exec_date", "exercise_date", "expiration_date"):
            d = pd.to_datetime(df[c].replace("", None))
            bad_dates += int(((d < "1900-01-01") | (d > "2200-12-31")).sum())
        bad_value += int((df["trans_value_usd"] < 0).sum())
        bad_ad += int((~df["trans_acq_disp"].isin(["A", "D"])).sum())
        for c in ("is_director", "is_officer", "is_ten_pct_owner", "is_other", "aff_10b5_1_filing",
                  "is_10b5_1_trans", "is_amendment", "is_superseded", "equity_swap"):
            bad_flags += int((~df[c].isin([0, 1])).sum())
        mn, mx = fd.min(), fd.max()
        min_fd = mn if min_fd is None else min(min_fd, mn)
        max_fd = mx if max_fd is None else max(max_fd, mx)
        tdv = td[ok_td]
        if len(tdv):
            min_td = tdv.min() if min_td is None else min(min_td, tdv.min())
            max_td = tdv.max() if max_td is None else max(max_td, tdv.max())
        for a in df["accession_number"].unique():
            if acc_quarters.setdefault(a, q) != q:
                acc_multi += 1
        del df

    hold_rows_q, hold_tables, hold_n = {}, Counter(), 0
    for p in hparts:
        h = pd.read_parquet(p, columns=["table"])
        hold_rows_q[p.stem] = len(h)
        hold_n += len(h)
        hold_tables.update(h["table"].value_counts().to_dict())

    print("=" * 78)
    print("INSIDER TRANSACTIONS VALIDATION")
    print("=" * 78)
    print(f"transactions: {n:,} rows in {len(parts)} quarters; holdings: {hold_n:,} rows")
    print(f"rows per table (transactions): {dict(rows_table)}")
    print(f"rows per table (holdings):     {dict(hold_tables)}")
    print(f"filing_date range: {min_fd.date()} .. {max_fd.date()}")
    print(f"trans_date range (excluding trans_date_suspect): {min_td.date()} .. {max_td.date()}")

    print("\nRows per quarter (transactions / holdings):")
    for q in sorted(rows_q):
        print(f"  {q}  {rows_q[q]:>9,}  {hold_rows_q.get(q, 0):>9,}")

    print("\nNull or empty rate per column (transactions):")
    for c in TRANS_COLUMNS:
        print(f"  {c:<24} {nulls[c] / n:7.2%}")

    print("\ntrans_code distribution:")
    for code, cnt in codes.most_common():
        print(f"  {code:<8} {cnt:>10,}  {cnt / n:6.2%}")

    print("\nFlags:")
    print(f"  is_10b5_1_trans share:       {n10b5 / n:.2%}")
    print(f"  is_late share (of known):    {nlate / max(nlate_known, 1):.2%}  (known for {nlate_known / n:.2%} of rows)")
    print(f"  trans_date_suspect rows:     {nsuspect:,}")
    print(f"  is_narrow_key_dup rows:      {nnarrow:,} ({nnarrow / n:.2%})")
    print(f"  is_superseded rows:          {nsuperseded:,} ({nsuperseded / n:.2%})")
    print(f"  amendment rows:              {namend:,}; with amended_accession: {namend_linked:,}"
          f" ({namend_linked / max(namend, 1):.1%})")

    print("\nAssertions:")
    check(columns_ok, "transaction columns match the required schema and order")
    check(dup_keys == 0, f"no duplicate (accession, owner, table, trans_sk) keys [{dup_keys}]")
    check(acc_multi == 0, f"each accession appears in one source_quarter only [{acc_multi}]")
    check(bad_iso == 0, f"all non-empty dates are ISO YYYY-MM-DD [{bad_iso}]")
    check(bad_dates == 0, "dates in range: filing_date 2006-01-01..today; trans_date 1990-01-01..filing_date "
          f"unless trans_date_suspect; other dates 1900..2200 [{bad_dates}]")
    check(bad_value == 0, f"trans_value_usd >= 0 [{bad_value}]")
    check(bad_ad == 0, f"trans_acq_disp in {{A, D}} [{bad_ad}]")
    check(bad_flags == 0, f"0/1 flag columns contain only 0/1 [{bad_flags}]")
    if not skip_csv_count:
        for path, expected in ((OUT_TRANS, n), (OUT_HOLD, hold_n)):
            got = csv_rows(path)
            check(got == expected, f"{path.name}[.gz] row count {got} == parquet {expected}")
    print(f"\n{'ALL ASSERTIONS PASSED' if not failures else f'{len(failures)} ASSERTION(S) FAILED'}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-csv-count", action="store_true")
    args = ap.parse_args(argv)
    buf = io.StringIO()
    with redirect_stdout(buf):
        failures = run(args.skip_csv_count)
    report = buf.getvalue()
    print(report, end="")
    REPORT.write_text(report)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
