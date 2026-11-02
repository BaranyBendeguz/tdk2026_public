# Select the opportunistic CFO purchases with opp_buy.py.
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from opp_buy import add_buy_patterns, pattern_report, purchase_mask  # noqa: E402

PARTS = HERE.parents[1] / "data" / "parquet" / "insider_transactions"


def load_purchases() -> pd.DataFrame:
    parts = sorted(PARTS.glob("*.parquet"))
    if not parts:
        sys.exit(f"no parquet partitions in {PARTS}; run src/build.py first")
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        frames.append(df[purchase_mask(df)])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = add_buy_patterns(load_purchases())  # all insiders' purchases, not only the CFO rows
    stats = {"all_purchases": pattern_report(p)}
    cfo = p[p["officer_title_clean"] == "cfo"]
    print("CFO:")
    stats["cfo_purchases"] = pattern_report(cfo)
    cfo_opp = cfo[cfo["is_opportunistic"]]
    stats["cfo_opp_rows"] = len(cfo_opp)
    stats["cfo_opp_buy_days"] = int(cfo_opp[["owner_cik", "issuer_cik", "trans_date"]].drop_duplicates().shape[0])
    print(f"cfo_opp.csv: {stats['cfo_opp_rows']:,} rows, {stats['cfo_opp_buy_days']:,} buy days")
    cfo_opp.to_csv(HERE / "cfo_opp.csv", index=False, date_format="%Y-%m-%d")
    (HERE / "opp_buy_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
