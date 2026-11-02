# Export one trained configuration into a self-contained folder.

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import data  # noqa: E402
import selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

DESCRIPTIONS = {
    "stk_ret_1d": "a részvény hozama az eseménynapon (1 tőzsdenap)",
    "stk_ret_1w": "a részvény 1 hetes (5 tőzsdenapos) hozama",
    "stk_ret_1m": "a részvény 1 hónapos (21 tőzsdenapos) hozama",
    "stk_ret_3m": "a részvény 3 hónapos (63 tőzsdenapos) hozama",
    "stk_ret_6m": "a részvény 6 hónapos (126 tőzsdenapos) hozama",
    "stk_ret_12m": "a részvény 12 hónapos (252 tőzsdenapos) hozama",
    "stk_vol_1m": "1 hónapos volatilitás (napi log-hozamok szórása, évesítve)",
    "stk_vol_3m": "3 hónapos volatilitás, évesítve",
    "stk_vol_12m": "12 hónapos volatilitás, évesítve",
    "stk_dist_52w_high": "záróár / 52 hetes csúcs − 1 (≤ 0)",
    "stk_dist_52w_low": "záróár / 52 hetes mélypont − 1 (≥ 0)",
    "stk_log_price": "log(nem igazított záróár), az ár, ahogy akkor kereskedtek",
    "stk_log_adv_1m": "log(1 + átlagos napi dollárforgalom az utolsó 21 napban)",
    "stk_volume_ratio": "az utolsó 5 nap átlagos forgalma / az utolsó 63 napé",
    "sector": "Yahoo-szektor (kategória)",
    "sec_ret_12m": "a szektor-ETF 12 hónapos hozama",
    "mkt_ret_6m": "a SPY 6 hónapos hozama",
    "rel_sec_ret_6m": "a részvény 6 hónapos hozama mínusz a szektor-ETF-é",
    "rel_sec_ret_12m": "a részvény 12 hónapos hozama mínusz a szektor-ETF-é",
    "macro_fed_funds": "Fed funds ráta (FRED DFF), a kötés előtti napon",
    "macro_curve_10y2y": "10 éves mínusz 2 éves kincstári hozam (FRED T10Y2Y)",
    "macro_credit_spread": "Moody's Baa hozam mínusz 10 éves kincstári hozam (FRED BAA10Y)",
    "macro_vix": "VIX (FRED VIXCLS), a kötés előtti napon",
    "macro_cpi_yoy": "CPI-infláció éves alapon, a kötés előtt publikált adat (ALFRED-vintage)",
    "macro_gdp_growth": "reál-GDP negyedéves növekedése évesítve, (q / q₋₁)⁴ − 1, a kötés előtt publikált adat (ALFRED-vintage)",
    "ins_n_insiders_30d": "hány különböző insider vett a részvényből az utolsó 30 napban (ő maga is)",
    "ins_log_value_30d": "log(1 + az összes insider-vásárlás dollárértéke az utolsó 30 napban)",
    "ins_days_since_prev": "napok az insider előző vásárlása óta ugyanabból a részvényből (első vételnél üres)",
    "log_value": "log(1 + a vásárlás dollárértéke; több kötési nap esetén összegezve)",
    "price_vs_close": "az insider átlagos vételi ára / a záróár az eseménynapon − 1",
    "x_frac_holdings_bought": "a direkt módon vett részvények / a direkt állomány a vétel után (1 = új pozíció)",
    "ret_vs_mkt_3m": "stk_ret_3m − mkt_ret_3m",
    "ret_vs_mkt_6m": "stk_ret_6m − mkt_ret_6m",
    "ret_vs_mkt_12m": "stk_ret_12m − mkt_ret_12m",
    "real_rate_3m": "3 hónapos T-bill hozam mínusz a CPI-infláció (reálkamat)",
    "mom_12_1": "12 hónapos momentum az utolsó hónap nélkül: (1 + ret_12m) / (1 + ret_1m) − 1",
    "vol_adj_ret_12m": "stk_ret_12m / stk_vol_12m",
    "cfo_share_of_insider_value_30d": "a vásárlás értéke / az összes insider-vásárlás értéke az utolsó 30 napban",
}
META = ["event_id", "owner_cik", "owner_name", "issuer_cik", "issuer_name", "symbol", "trans_date", "signal_date",
        "entry_date", "exit_date"]
TARGETS = [data.TARGET, "stock_ret", "spy_ret"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "data" / "model" / "xgb_train2016")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: modell/xgb_2016_{n}feat")
    args = ap.parse_args(argv)
    report = json.loads((args.model_dir / "report.json").read_text())
    features = report["final_features"]
    train_start = pd.Timestamp(report["train_start"])
    test_start = pd.Timestamp(report["test_start"])
    out = args.out or ROOT / "modell" / f"xgb_2016_{len(features)}feat"
    (out / "backtest").mkdir(parents=True, exist_ok=True)

    df, _, counts = data.load()
    df = df[df["signal_date"] >= train_start].reset_index(drop=True)
    tr, te = cv.test_split(df, test_start, train_start)
    df["split"] = "purged (target window reaches the test period)"
    df.loc[tr, "split"] = "train"
    df.loc[te, "split"] = "test"
    test_pred = pd.read_parquet(args.model_dir / "test_predictions.parquet")[["event_id", "pred"]]
    cv_pred = pd.read_parquet(args.model_dir / "cv_predictions.parquet")[["event_id", "fold", "pred"]]
    df = (df.merge(test_pred.rename(columns={"pred": "pred_test"}), on="event_id", how="left")
          .merge(cv_pred.rename(columns={"pred": "pred_cv", "fold": "cv_fold"}), on="event_id", how="left"))

    booster = xgb.Booster()
    booster.load_model(args.model_dir / "model.json")
    check = booster.predict(xgb.DMatrix(df.loc[df["split"] == "test", features], enable_categorical=True))
    if not np.allclose(check, df.loc[df["split"] == "test", "pred_test"], atol=1e-6):
        raise RuntimeError("model.json does not reproduce the saved test predictions")
    shutil.copy(args.model_dir / "model.json", out / "model.json")

    table = df[META + ["split", "cv_fold"] + features + TARGETS + ["pred_test", "pred_cv"]]
    table.to_csv(out / f"features_{len(features)}.csv", index=False, date_format="%Y-%m-%d")
    table.to_parquet(out / f"features_{len(features)}.parquet", index=False)
    df.loc[df["split"] == "test", META + TARGETS + ["pred_test"]].rename(columns={"pred_test": "pred"}).to_csv(
        out / "test_predictions.csv", index=False, date_format="%Y-%m-%d")

    groups = selection.feature_groups(features)
    imp = pd.read_csv(args.model_dir / "feature_importance.csv", index_col=0)
    flist = pd.DataFrame({
        "feature": features,
        "group": [groups[f] for f in features],
        "source": ["engineered (modell/xgb/data.py)" if f in data.ENGINEERED else "feature table (docs/features.md)"
                   for f in features],
        "description": [DESCRIPTIONS[f] for f in features],
    }).join(imp[["perm_ic_drop_mean", "perm_positive_fold_share", "mean_abs_shap_test"]], on="feature")
    flist.to_csv(out / "feature_list.csv", index=False)

    for f in sorted((args.model_dir / "backtest").glob("*")):
        shutil.copy(f, out / "backtest" / f.name)
    bt = json.loads((args.model_dir / "backtest" / "backtest_top5_cost0.json").read_text())
    stage = report[f"stage_{report['chosen_feature_set']}"]
    config = {
        "name": f"B: XGBoost, training from 2016, {len(features)} features",
        "source_run": str(args.model_dir.relative_to(ROOT)),
        "reproduce": "python modell/xgb/train.py --trials 100 --train-start 2016-01-01 --val-years 2018-2021 "
                     "--out-dir data/model/xgb_train2016 && python modell/xgb/backtest.py "
                     "--model-dir data/model/xgb_train2016 --top 5 --cost-bps 0",
        "target": "excess_ret_6m: 126 market days total return of the stock minus SPY, entry at the close of the "
                  "first market day after signal_date",
        "sample": {**counts, "rules": "Yahoo symbol, realised target, a Table I line, price at the anchor >= $1, "
                                     "one row per (owner, issuer, signal_date)"},
        "train_start": report["train_start"], "test_start": report["test_start"],
        "final_train": {"n": report["n_final_train"], "signal_range": report["final_train_signal_range"]},
        "n_test": report["n_test"],
        "validation": {"folds": report["folds"], "metric": "pooled Spearman IC per validation year, mean",
                       "purging": "training signals need exit_date < start of the validation year"},
        "winsor_bounds_final": report["winsor_bounds_final"],
        "params": report["final_params"],
        "xgboost_fixed": {"tree_method": "hist", "enable_categorical": True, "random_state": 42},
        "features": features,
        "categories": {c: [str(v) for v in df[c].cat.categories] for c in data.CATEGORICAL if c in features},
        "prefilter_dropped": report["prefilter_dropped"],
        "selection_dropped": report["selection_dropped"],
        "cv": {"ic_mean": stage["cv_ic"], "ic_std": stage["cv_ic_std"], "ic_by_year": stage["cv_ic_by_year"]},
        "test": report["test"],
        "test_ic_within_month_mean": float(cv.ic_within_month(df.loc[te, "pred_test"], df.loc[te, data.TARGET],
                                                              df.loc[te, "signal_date"])),
        "backtest_top5": {"period": bt["period"], "cohorts": bt["n_cohorts"],
                          "cost_0bps": bt["strategies"], "random_top5_cost_0bps": bt["random_top_n"]},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out}: {len(table)} signals ({(table['split'] == 'train').sum()} train, "
          f"{(table['split'] == 'test').sum()} test), {len(features)} features; model.json reproduces the test "
          f"predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
