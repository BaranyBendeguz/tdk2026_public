# Feature selection for the monthly top-N strategy.

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import data  # noqa: E402
import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "model" / "xgb_select"
TRAIN_START = pd.Timestamp("2016-01-01")
VAL_YEARS = (2018, 2021)
SEEDS = (42, 7, 2026)  # with --seed s other than train.SEED: (s, s + 1000, s + 2000)
CLUSTER_RHO = 0.8
PERM_REPEATS = 2

GROUP_RULES = [
    ("stock returns", lambda f: f.startswith(("stk_ret_", "ret_vs_mkt_", "vol_adj_ret_")) or f == "mom_12_1"),
    ("stock risk", lambda f: f.startswith("stk_vol_") or f in ("stk_downside_vol_3m", "stk_beta_12m",
                                                               "downside_share_3m", "vol_trend")),
    ("price and liquidity", lambda f: f in ("stk_dist_52w_high", "stk_dist_52w_low", "stk_log_price",
                                            "stk_log_adv_1m", "stk_volume_ratio", "stk_days_since_price")),
    ("sector and market", lambda f: f == "sector" or f.startswith(("sec_ret_", "mkt_ret_", "rel_sec_ret_"))),
    ("macro", lambda f: f.startswith("macro_") or f == "real_rate_3m"),
    ("insider activity", lambda f: f.startswith("ins_") or f == "cfo_share_of_insider_value_30d"),
    ("insider role", lambda f: f.startswith("role_") or f in ("insider_type", "exec_seniority", "n_roles",
                                                             "title_missing", "insider_is_entity")),
    ("trade", lambda f: f in ("log_value", "n_fills", "price_vs_close", "trade_value_to_adv",
                              "x_frac_holdings_bought", "x_filing_lag_days", "n_trade_days", "signal_delay_days")),
]
INSIDER_GROUPS = ("insider activity", "insider role", "trade")

log = logging.getLogger("xgb.select")


def feature_groups(features: list[str]) -> dict[str, str]:
    out = {f: next((g for g, rule in GROUP_RULES if rule(f)), None) for f in features}
    missing = [f for f, g in out.items() if g is None]
    if missing:
        raise ValueError(f"features without a group: {missing}")
    return out


def wm_ic(pred: np.ndarray, df: pd.DataFrame, va: np.ndarray) -> float:
    return cv.ic_within_month(pred, df[data.TARGET].iloc[va], df["signal_date"].iloc[va])


def cv_by_fold(params: dict, df: pd.DataFrame, features: list[str], folds) -> np.ndarray:
    """Within-month IC per fold, averaged over SEEDS."""
    out = np.zeros(len(folds))
    for seed in SEEDS:
        p = {**params, "random_state": seed}
        for k, (_, tr, va) in enumerate(folds):
            model, _, _ = train.fit(p, df, features, tr)
            out[k] += wm_ic(model.predict(df[features].iloc[va]), df, va) / len(SEEDS)
    return out


def group_ablation(params: dict, df: pd.DataFrame, features: list[str], groups: dict[str, str], folds) -> pd.DataFrame:
    base = cv_by_fold(params, df, features, folds)
    variants = [(g, (g,)) for g in dict.fromkeys(groups.values())]
    variants.append(("insider-specific (activity + role + trade)", INSIDER_GROUPS))
    rows = [{"group": "all features", "n_features": len(features), "cv_ic_without": np.nan,
             "delta_ic": np.nan, "delta_se": np.nan, "cv_ic_only": float(base.mean())}]
    for name, gs in variants:
        members = [f for f in features if groups[f] in gs]
        without = cv_by_fold(params, df, [f for f in features if f not in members], folds)
        only = cv_by_fold(params, df, members, folds)
        d = base - without  # positive: the group helps
        rows.append({"group": name, "n_features": len(members), "cv_ic_without": float(without.mean()),
                     "delta_ic": float(d.mean()), "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                     "cv_ic_only": float(only.mean())})
        log.info("ablation %-45s n=%2d  without %.4f  delta %+.4f  only %.4f", name, len(members),
                 without.mean(), d.mean(), only.mean())
    return pd.DataFrame(rows)


def clusters_of(dev: pd.DataFrame, features: list[str], rho: float = CLUSTER_RHO) -> list[list[str]]:
    """Connected components of the |Spearman| > rho graph; categorical features stay alone."""
    num = [f for f in features if f not in data.CATEGORICAL]
    corr = dev[num].corr(method="spearman").abs()
    parent = {f: f for f in features}

    def root(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    for i, a in enumerate(num):
        for b in num[i + 1:]:
            if corr.loc[a, b] > rho:
                parent[root(b)] = root(a)
    comps: dict[str, list[str]] = {}
    for f in features:
        comps.setdefault(root(f), []).append(f)
    return list(comps.values())


def _shuffle(x: pd.DataFrame, cols, rng) -> pd.DataFrame:
    """Columns `cols` of x with their rows shuffled together (dtype kept)."""
    return x[cols].iloc[rng.permutation(len(x))].set_axis(x.index)


def shadow_selection(params: dict, df: pd.DataFrame, features: list[str], clusters: list[list[str]],
                     folds) -> tuple[pd.DataFrame, float, list[str]]:
    rng = np.random.default_rng(train.SEED)
    shadows = {f: f"shadow__{f}" for f in features}
    rows = []
    for seed in SEEDS:
        p = {**params, "random_state": seed}
        for label, tr, va in folds:
            x = df[features]
            xtr = pd.concat([x.iloc[tr], _shuffle(x.iloc[tr], features, rng).rename(columns=shadows)], axis=1)
            xva = pd.concat([x.iloc[va], _shuffle(x.iloc[va], features, rng).rename(columns=shadows)], axis=1)
            y = df[data.TARGET].iloc[tr]
            model = train.make_model(p).fit(xtr, y.clip(*cv.winsor_bounds(y)))
            base = wm_ic(model.predict(xva), df, va)
            items = [("real", " + ".join(c), c) for c in clusters] + [("shadow", s, [s]) for s in shadows.values()]
            for kind, name, cols in items:
                drops = []
                for _ in range(PERM_REPEATS):
                    xp = xva.copy()
                    xp[cols] = _shuffle(xva, cols, rng)
                    drops.append(base - wm_ic(model.predict(xp), df, va))
                rows.append({"seed": seed, "fold": label, "kind": kind, "item": name, "imp": float(np.mean(drops))})
        log.info("shadow selection: seed %d done", seed)
    r = pd.DataFrame(rows)
    shadow_max_round = r[r["kind"] == "shadow"].groupby(["seed", "fold"])["imp"].max().rename("shadow_max")
    r = r.join(shadow_max_round, on=["seed", "fold"])
    table = r.groupby(["kind", "item"])["imp"].agg(mean_imp="mean", se_imp=lambda s: s.std() / np.sqrt(len(s)))
    # diagnostic: share of rounds (seed x fold) in which the item beats that round's best shadow
    hits = r.assign(hit=r["imp"] > r["shadow_max"]).groupby(["kind", "item"])["hit"].mean()
    table["share_rounds_beating_shadow_max"] = hits
    shadow_means = table.loc["shadow", "mean_imp"]
    real = table.loc["real"]
    rule, threshold = "beats the best shadow", float(shadow_means.max())
    if not (real["mean_imp"] > threshold).any():  # fallback fixed in advance: 95th percentile of the shadows
        rule, threshold = "beats the 95th percentile of the shadows", float(shadow_means.quantile(0.95))
    kept_items = real.index[real["mean_imp"] > threshold].tolist()
    table["kept"] = [k == "real" and i in kept_items for k, i in table.index]
    kept = [f for c in clusters if " + ".join(c) in kept_items for f in c]
    log.info("shadow rule: %s", rule)
    return table.sort_values("mean_imp", ascending=False).reset_index(), threshold, kept


def load_stage(out_dir: Path, name: str, n_features: int, folds) -> dict:
    """The result of train.tune() rebuilt from its saved trials_{name}.csv."""
    t = pd.read_csv(out_dir / f"trials_{name}.csv")
    best = t[t["state"] == "COMPLETE"].sort_values("value", ascending=False).iloc[0]
    params = {k.removeprefix("params_"): v for k, v in best.items() if k.startswith("params_") and pd.notna(v)}
    for k in ("n_estimators", "max_depth"):
        params[k] = int(params[k])
    by_fold = ast.literal_eval(best["user_attrs_ic_by_year"])
    return {"params": params, "cv_ic": float(best["value"]), "cv_ic_std": float(best["user_attrs_ic_std"]),
            "cv_ic_by_year": dict(zip([y for y, _, _ in folds], by_fold)), "n_features": n_features,
            "trials": len(t), "pruned": int((t["state"] == "PRUNED").sum())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--resume", action="store_true",
                    help="reuse the saved base tuning, ablation and shadow selection of --out-dir (steps 2-4)")
    ap.add_argument("--seed", type=int, default=train.SEED, help="seed of Optuna, XGBoost and the permutations")
    args = ap.parse_args(argv)
    global SEEDS
    if args.seed != train.SEED:
        SEEDS = (args.seed, args.seed + 1000, args.seed + 2000)
        train.SEED = args.seed
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, features, counts = data.load()
    dev = df[(df["signal_date"] >= TRAIN_START) & (df["signal_date"] < cv.TEST_START)]
    folds = cv.halfyear_folds(df, *VAL_YEARS, TRAIN_START)
    log.info("folds (half year, train, validation): %s", [(y, len(a), len(b)) for y, a, b in folds])

    pre, dropped = train.prefilter(dev, features)
    groups = feature_groups(pre)
    clusters = clusters_of(dev, pre)
    if args.resume:
        base = load_stage(out_dir, "all", len(pre), folds)
        ablation = pd.read_csv(out_dir / "group_ablation.csv")
        shadow = pd.read_csv(out_dir / "shadow_importance.csv")
        threshold = float(shadow.loc[shadow["kind"] == "shadow", "mean_imp"].max())
        kept_items = set(shadow.loc[shadow["kept"], "item"])
        kept = [f for c in clusters if " + ".join(c) in kept_items for f in c]
        log.info("resumed steps 2-4 from %s", out_dir)
    else:
        base = train.tune(df, pre, folds, args.trials, "all", out_dir, score=wm_ic)
        ablation = group_ablation(base["params"], df, pre, groups, folds)
        ablation.to_csv(out_dir / "group_ablation.csv", index=False)
        shadow, threshold, kept = shadow_selection(base["params"], df, pre, clusters, folds)
        shadow.to_csv(out_dir / "shadow_importance.csv", index=False)
    log.info("shadow selection: %d of %d features kept (threshold = best shadow mean %.4f)", len(kept), len(pre),
             threshold)

    final = train.tune(df, kept, folds, args.trials, "selected", out_dir, score=wm_ic)
    oof = []
    for label, tr, va in folds:
        model, _, _ = train.fit(final["params"], df, kept, tr)
        oof.append(df.iloc[va][["event_id", "signal_date", data.TARGET]].assign(
            fold=label, pred=model.predict(df[kept].iloc[va])))
    pd.concat(oof).to_parquet(out_dir / "cv_predictions.parquet", index=False)

    tr, te = cv.test_split(df, cv.TEST_START, TRAIN_START)
    model, bounds, train_mean = train.fit(final["params"], df, kept, tr)
    pred = model.predict(df[kept].iloc[te])
    test = cv.evaluate(pred, df[data.TARGET].iloc[te], df["signal_date"].iloc[te], bounds, train_mean)
    model.save_model(out_dir / "model.json")
    df.iloc[te][["event_id", "owner_cik", "issuer_cik", "symbol", "trans_date", "signal_date", "entry_date",
                 "exit_date", "stock_ret", "spy_ret", data.TARGET]].assign(pred=pred).to_parquet(
        out_dir / "test_predictions.parquet", index=False)

    report = {
        "sample": counts, "train_start": str(TRAIN_START.date()), "test_start": str(cv.TEST_START.date()),
        "metric": "within-month Spearman IC", "folds": [{"fold": y, "n_train": len(a), "n_val": len(b)}
                                                       for y, a, b in folds],
        "n_final_train": int(len(tr)), "n_test": int(len(te)),
        "prefilter_dropped": dropped, "features_after_prefilter": pre, "groups": groups,
        "base": base, "group_ablation": ablation.to_dict(orient="records"),
        "clusters": [c for c in clusters if len(c) > 1], "shadow_threshold": threshold,
        "kept_features": kept, "dropped_by_shadow": [f for f in pre if f not in kept],
        "final": final, "test": test, "seed": train.SEED, "seeds": list(SEEDS), "runtime_s": round(time.time() - t0),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("base CV %.4f (%d features) | final CV %.4f (%d features) | test within-month IC %.4f, pooled IC %.4f",
             base["cv_ic"], len(pre), final["cv_ic"], len(kept), test["ic_within_month_mean"], test["ic"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
