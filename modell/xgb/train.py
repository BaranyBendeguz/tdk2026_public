# XGBoost model for the 6-month excess return (over SPY) of the opportunistic CFO purchases.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import data  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "model" / "xgb"
SEED = 42
N_JOBS = int(os.environ.get("XGB_N_JOBS", os.cpu_count() or 4))  # threads per model
NEAR_CONSTANT = 0.995
CORR_MAX = 0.95
PERM_REPEATS = 3
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000, 10000]

log = logging.getLogger("xgb")


def prefilter(dev: pd.DataFrame, features: list[str]) -> tuple[list[str], dict[str, str]]:
    dropped, kept = {}, []
    for f in features:
        top = dev[f].value_counts(normalize=True, dropna=False).iloc[0]
        if dev[f].nunique() <= 1 or top > NEAR_CONSTANT:
            dropped[f] = f"near-constant: one value in {top:.1%} of rows"
        else:
            kept.append(f)
    num = [f for f in kept if f not in data.CATEGORICAL]
    corr = dev[num].corr(method="spearman").abs()
    out = []
    for f in kept:
        twin = next((k for k in out if f in num and k in num and corr.loc[f, k] > CORR_MAX), None)
        if twin:
            dropped[f] = f"|Spearman| {corr.loc[f, twin]:.3f} with {twin}"
        else:
            out.append(f)
    return out, dropped


def make_model(params: dict) -> xgb.XGBRegressor:
    """params may set random_state; the default seed is SEED."""
    return xgb.XGBRegressor(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, verbosity=0,
                            **{"random_state": SEED, **params})


def fit(params: dict, df: pd.DataFrame, features: list[str], idx: np.ndarray):
    y = df[data.TARGET].iloc[idx]
    bounds = cv.winsor_bounds(y)
    model = make_model(params).fit(df[features].iloc[idx], y.clip(*bounds))
    return model, bounds, float(y.clip(*bounds).mean())


def suggest(trial: optuna.Trial) -> dict:
    p = {
        "objective": trial.suggest_categorical(
            "objective", ["reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }
    if p["objective"] == "reg:pseudohubererror":
        p["huber_slope"] = trial.suggest_float("huber_slope", 0.05, 1.0, log=True)
    return p


def pooled_ic(pred: np.ndarray, df: pd.DataFrame, va: np.ndarray) -> float:
    return cv.ic(pred, df[data.TARGET].iloc[va])


def tune(df: pd.DataFrame, features: list[str], folds, n_trials: int, name: str, out_dir: Path,
         score=pooled_ic) -> dict:
    """score(pred, df, validation positions) -> float is maximised, averaged over the folds."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        scores = []
        for k, (_, tr, va) in enumerate(folds):
            model, _, _ = fit(params, df, features, tr)
            scores.append(score(model.predict(df[features].iloc[va]), df, va))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("ic_by_year", [round(s, 4) for s in scores])
        trial.set_user_attr("ic_std", float(np.std(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=3))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(out_dir / f"trials_{name}.csv", index=False)
    best = study.best_trial
    log.info("tune %s: best mean IC %.4f (std %.4f) after %d trials in %.0fs", name, best.value,
             best.user_attrs["ic_std"], len(study.trials), time.time() - t0)
    return {"params": best.params, "cv_ic": best.value, "cv_ic_std": best.user_attrs["ic_std"],
            "cv_ic_by_year": dict(zip([y for y, _, _ in folds], best.user_attrs["ic_by_year"])),
            "n_features": len(features), "trials": len(study.trials),
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}


def permutation_importance(params: dict, df: pd.DataFrame, features: list[str], folds) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for year, tr, va in folds:
        model, _, _ = fit(params, df, features, tr)
        xv, yv = df[features].iloc[va], df[data.TARGET].iloc[va]
        base = cv.ic(model.predict(xv), yv)
        for f in features:
            drops = []
            for _ in range(PERM_REPEATS):
                xp = xv.copy()
                xp[f] = xv[f].iloc[rng.permutation(len(xv))].set_axis(xv.index)
                drops.append(base - cv.ic(model.predict(xp), yv))
            rows.append({"year": year, "feature": f, "ic_drop": float(np.mean(drops))})
    imp = pd.DataFrame(rows).groupby("feature")["ic_drop"].agg(["mean", "std", lambda s: (s > 0).mean()])
    imp.columns = ["perm_ic_drop_mean", "perm_ic_drop_std", "perm_positive_fold_share"]
    return imp.sort_values("perm_ic_drop_mean", ascending=False)


def ridge_design(df: pd.DataFrame, features: list[str], tr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Clip at the training 1% / 99% quantiles, impute the training median, standardise, one-hot categoricals."""
    num = [f for f in features if f not in data.CATEGORICAL]
    xt = df[num].iloc[tr]
    lo, hi = xt.quantile(0.01), xt.quantile(0.99)
    xt = xt.clip(lo, hi, axis=1)
    med = xt.median()
    xt = xt.fillna(med)
    mu, sd = xt.mean(), xt.std().replace(0, 1)
    x = ((df[num].iloc[idx].clip(lo, hi, axis=1).fillna(med) - mu) / sd).fillna(0)
    cats = [pd.get_dummies(df[c].iloc[idx].astype(str), prefix=c).reindex(
        columns=[f"{c}_{v}" for v in df[c].iloc[tr].astype(str).unique()], fill_value=0)
        for c in features if c in data.CATEGORICAL]
    return pd.concat([x] + cats, axis=1).to_numpy(float)


def ridge_baseline(df: pd.DataFrame, features: list[str], folds, test: tuple[np.ndarray, np.ndarray]) -> dict:
    def run(alpha, tr, idx):
        y = df[data.TARGET].iloc[tr]
        b = cv.winsor_bounds(y)
        m = Ridge(alpha=alpha).fit(ridge_design(df, features, tr, tr), y.clip(*b))
        return m.predict(ridge_design(df, features, tr, idx)), b, float(y.clip(*b).mean())

    scores = {a: float(np.mean([cv.ic(run(a, tr, va)[0], df[data.TARGET].iloc[va]) for _, tr, va in folds]))
              for a in RIDGE_ALPHAS}
    alpha = max(scores, key=scores.get)
    tr, te = test
    pred, b, mean = run(alpha, tr, te)
    return {"alpha": alpha, "cv_ic_by_alpha": scores, "cv_ic": scores[alpha],
            "test": cv.evaluate(pred, df[data.TARGET].iloc[te], df["signal_date"].iloc[te], b, mean)}


def main(argv: list[str] | None = None) -> int:
    global SEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=100, help="Optuna trials per tuning stage")
    ap.add_argument("--min-price", type=float, default=data.MIN_PRICE)
    ap.add_argument("--train-start", type=pd.Timestamp, default=None,
                    help="first signal_date used for training (default: all history)")
    ap.add_argument("--test-start", type=pd.Timestamp, default=cv.TEST_START)
    ap.add_argument("--val-years", default=f"{cv.VAL_YEARS[0]}-{cv.VAL_YEARS[-1]}",
                    help="walk-forward validation years, e.g. 2018-2021")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED, help="seed of Optuna, XGBoost and the permutations")
    args = ap.parse_args(argv)
    SEED = args.seed
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    first, last = (int(y) for y in args.val_years.split("-"))
    val_years = list(range(first, last + 1))
    if val_years[-1] >= args.test_start.year:
        ap.error("validation years must end before the test period")
    df, features, counts = data.load(args.min_price)
    dev = df[(df["signal_date"] < args.test_start)
             & (df["signal_date"] >= (args.train_start or df["signal_date"].min()))]
    folds = cv.folds(df, val_years, args.train_start)
    test = cv.test_split(df, args.test_start, args.train_start)
    log.info("sample: %s; development %d, test %d signals", counts, len(dev), len(test[1]))
    log.info("folds (year, train, validation): %s", [(y, len(tr), len(va)) for y, tr, va in folds])

    pre, dropped = prefilter(dev, features)
    log.info("prefilter: %d -> %d features", len(features), len(pre))
    stage_all = tune(df, pre, folds, args.trials, "all", out_dir)

    imp = permutation_importance(stage_all["params"], df, pre, folds)
    selected = [f for f in pre if imp.loc[f, "perm_ic_drop_mean"] > 0]
    log.info("selection: %d of %d features have positive permutation importance", len(selected), len(pre))
    stage_sel = tune(df, selected, folds, args.trials, "selected", out_dir)
    chosen = "selected" if stage_sel["cv_ic"] > stage_all["cv_ic"] else "all"
    best = stage_sel if chosen == "selected" else stage_all
    final_features = selected if chosen == "selected" else pre
    log.info("chosen feature set: %s (%d features, CV IC %.4f)", chosen, len(final_features), best["cv_ic"])

    # out-of-fold predictions of the chosen configuration, for the record
    oof = []
    for year, tr, va in folds:
        model, _, _ = fit(best["params"], df, final_features, tr)
        oof.append(df.iloc[va][["event_id", "signal_date", data.TARGET]].assign(
            fold=year, pred=model.predict(df[final_features].iloc[va])))
    pd.concat(oof).to_parquet(out_dir / "cv_predictions.parquet", index=False)

    tr, te = test
    model, bounds, train_mean = fit(best["params"], df, final_features, tr)
    pred = model.predict(df[final_features].iloc[te])
    test_metrics = cv.evaluate(pred, df[data.TARGET].iloc[te], df["signal_date"].iloc[te], bounds, train_mean)
    model.save_model(out_dir / "model.json")
    out = df.iloc[te][["event_id", "owner_cik", "issuer_cik", "symbol", "trans_date", "signal_date", "entry_date",
                       "exit_date", "stock_ret", "spy_ret", data.TARGET]].assign(pred=pred)
    out.to_parquet(out_dir / "test_predictions.parquet", index=False)

    contribs = model.get_booster().predict(
        xgb.DMatrix(df[final_features].iloc[te], enable_categorical=True), pred_contribs=True)
    shap = pd.Series(np.abs(contribs[:, :-1]).mean(axis=0), index=final_features, name="mean_abs_shap_test")
    imp.join(shap, how="outer").assign(selected=lambda x: x.index.isin(final_features)).sort_values(
        "perm_ic_drop_mean", ascending=False).to_csv(out_dir / "feature_importance.csv")

    ridge = ridge_baseline(df, final_features, folds, test)
    report = {
        "sample": counts, "n_development": len(dev), "n_test": int(len(te)), "n_final_train": int(len(tr)),
        "train_start": str(args.train_start.date()) if args.train_start is not None else None,
        "final_train_signal_range": [str(df["signal_date"].iloc[tr].min().date()),
                                     str(df["signal_date"].iloc[tr].max().date())],
        "test_start": str(args.test_start.date()), "validation_years": val_years,
        "folds": [{"year": y, "n_train": len(a), "n_val": len(b)} for y, a, b in folds],
        "features_input": len(features), "prefilter_dropped": dropped, "features_after_prefilter": len(pre),
        "stage_all": stage_all, "stage_selected": stage_sel,
        "selection_dropped": [f for f in pre if f not in selected], "chosen_feature_set": chosen,
        "final_features": final_features, "final_params": best["params"], "winsor_bounds_final": bounds,
        "test": test_metrics, "ridge_baseline": ridge, "seed": SEED, "runtime_s": round(time.time() - t0),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("test: IC %.4f, top quintile mean excess %.4f vs all %.4f (ridge test IC %.4f)", test_metrics["ic"],
             test_metrics["mean_excess_top_quintile"], test_metrics["mean_excess_all"], ridge["test"]["ic"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
