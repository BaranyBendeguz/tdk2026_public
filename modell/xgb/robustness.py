# Multi-seed robustness of configs B and C, and the model against simple rules.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "modell" / "roles"))
import backtest as bt  # noqa: E402
from role_backtest import newey_west  # noqa: E402

MODEL = ROOT / "data" / "model"
OUT = MODEL / "robust"
MAIN = {"B": MODEL / "xgb_train2016", "C": MODEL / "xgb_select"}
PORTFOLIOS = [(5, 0), (10, 0), (20, 0), (5, 25)]


def runs(cfg: str) -> list[tuple[int, Path]]:
    out = [(42, MAIN[cfg])]
    for d in sorted(OUT.glob(f"{cfg}_seed*")):
        if (d / "report.json").exists():
            out.append((int(d.name.split("seed")[1]), d))
    return out


def evaluate_run(d: Path, top: int, cost: float) -> dict | None:
    tag = f"top{top}_cost{cost:g}"
    jf, cf = d / "backtest" / f"backtest_{tag}.json", d / "backtest" / f"cohorts_{tag}.csv"
    if not jf.exists():
        return None
    r = json.loads(jf.read_text())
    s = r["strategies"]
    co = pd.read_csv(cf).pivot(index="month", columns="strategy", values="excess")
    model = co[f"model top {top}"].to_numpy()
    row = {"top": top, "cost_bps": cost,
           "cagr": s[f"model top {top}"]["cagr"], "sharpe": s[f"model top {top}"]["sharpe"],
           "spy_sharpe": s["SPY buy and hold"]["sharpe"], "all_sharpe": s["all opportunistic buys"]["sharpe"],
           "lowvol_sharpe": s[f"lowest volatility {top}"]["sharpe"],
           "random_sharpe_percentile": r["random_top_n"]["sharpe"]["model_percentile"]}
    for name, base in (("spy", 0.0), ("all", co["all opportunistic buys"].to_numpy()),
                       ("lowvol", co[f"lowest volatility {top}"].to_numpy())):
        nw = newey_west(model - base)
        row[f"vs_{name}_mean"], row[f"vs_{name}_t"] = nw["mean"], nw["t"]
    # concentration: the cohort mean excess without the two best cohorts
    row["vs_spy_mean_without_best2"] = float(np.sort(model)[:-2].mean())
    return row


def ensemble(cfg: str, members: list[tuple[int, Path]]) -> Path:
    preds = []
    for seed, d in members:
        p = pd.read_parquet(d / "test_predictions.parquet")
        month = p["signal_date"].dt.to_period("M")
        preds.append(p["pred"].groupby(month).rank(pct=True).rename(seed))
    base = pd.read_parquet(members[0][1] / "test_predictions.parquet")
    for (_, d), s in zip(members, preds):
        other = pd.read_parquet(d / "test_predictions.parquet")["event_id"]
        if not other.equals(base["event_id"]):
            raise ValueError(f"{d}: test signals differ from {members[0][1]}")
    out = OUT / f"{cfg}_ensemble"
    out.mkdir(parents=True, exist_ok=True)
    base.assign(pred=pd.concat(preds, axis=1).mean(axis=1)).to_parquet(out / "test_predictions.parquet", index=False)
    (out / "report.json").write_text(json.dumps({"members": [str(d.relative_to(ROOT)) for _, d in members]}, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200, help="random portfolios in the ensemble backtests")
    args = ap.parse_args(argv)
    rows, kept = [], []
    for cfg in ("B", "C"):
        members = runs(cfg)
        ens = ensemble(cfg, members)
        for top, cost in PORTFOLIOS:  # always rerun: the members change as seeds are added
            bt.main(["--model-dir", str(ens), "--top", str(top), "--cost-bps", str(cost), "--draws", str(args.draws)])
        for seed, d in members + [("ensemble", ens)]:
            rep = json.loads((d / "report.json").read_text())
            if cfg == "C" and seed != "ensemble":
                kept.append(pd.Series(1, index=rep["kept_features"], name=seed))
            test = rep.get("test", {})
            for top, cost in PORTFOLIOS:
                r = evaluate_run(d, top, cost)
                if r:
                    rows.append({"config": cfg, "seed": seed, "test_ic_within_month": test.get("ic_within_month_mean"),
                                 "n_features": len(rep.get("final_features", rep.get("kept_features", []))) or None,
                                 **r})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs.csv", index=False)

    seeds = df[df["seed"] != "ensemble"]
    g = seeds.groupby(["config", "top", "cost_bps"])
    summary = pd.concat({
        "n_seeds": g.size(),
        **{f"{c}_{q}": getattr(g[c], q)() for c in ("cagr", "sharpe", "vs_spy_t", "vs_all_t", "vs_lowvol_t")
           for q in ("median", "min", "max")},
        "share_sharpe_above_spy": g.apply(lambda x: (x["sharpe"] > x["spy_sharpe"]).mean()),
        "share_vs_spy_t_above_2": g["vs_spy_t"].apply(lambda s: (s > 2).mean()),
        "share_vs_all_t_above_2": g["vs_all_t"].apply(lambda s: (s > 2).mean()),
        "share_vs_lowvol_t_above_2": g["vs_lowvol_t"].apply(lambda s: (s > 2).mean()),
        "random_percentile_median": g["random_sharpe_percentile"].median(),
    }, axis=1)
    summary.to_csv(OUT / "summary.csv")
    freq = pd.concat(kept, axis=1).fillna(0)
    freq = freq.mean(axis=1).sort_values(ascending=False).rename("share_of_seeds_kept")
    freq.to_csv(OUT / "c_feature_frequency.csv")

    pd.set_option("display.width", 250)
    cols = ["config", "seed", "top", "cost_bps", "cagr", "sharpe", "random_sharpe_percentile", "vs_spy_mean",
            "vs_spy_t", "vs_all_mean", "vs_all_t", "vs_lowvol_mean", "vs_lowvol_t", "vs_spy_mean_without_best2"]
    print(df[cols].round(3).to_string(index=False))
    print("\n", summary.round(3).T.to_string())
    print("\nC: share of seeds that keep each feature\n", freq.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
