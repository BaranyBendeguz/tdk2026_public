# Figures for the thesis in black, white and grays only.

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modell" / "xgb"))
sys.path.insert(0, str(ROOT / "modell" / "features"))
import data  # noqa: E402
import factor_event as fe  # noqa: E402
import sources  # noqa: E402
import stock  # noqa: E402
from robustness import newey_west  # noqa: E402

OUT = ROOT / "figures"
BLACK, DARK, MID, LIGHT, PALE = "#000000", "#4d4d4d", "#808080", "#b3b3b3", "#e0e0e0"
WIDTH = 6.3  # inches, the text width of an A4 page with 2.5 cm margins
EVENT_DAYS = 130
TEST_START = pd.Timestamp("2022-01-01")

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.edgecolor": DARK, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": PALE, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.color": DARK, "ytick.color": DARK, "axes.labelcolor": BLACK, "text.color": BLACK,
    "lines.linewidth": 1.4, "figure.dpi": 100, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "text.parse_math": False,  # "$" is a currency sign here, not math
})


def comma(x: float, decimals: int = 0) -> str:
    """Number as text, with a decimal comma and a typographic minus."""
    return f"{x:.{decimals}f}".replace(".", ",").replace("-", "−")


def pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = FuncFormatter(lambda v, _: comma(v * 100, decimals) + "%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def num_axis(ax, axis: str = "y", decimals: int = 1) -> None:
    fmt = FuncFormatter(lambda v, _: comma(v, decimals))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"figures/{name}.pdf, .png")


def end_label(ax, x, y, text: str, dy: float = 0.0) -> None:
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8.5)


# 1. Price coverage by year ---------------------------------------------------------------------------------------
def fig_coverage() -> None:
    by = pd.DataFrame(json.loads((ROOT / "data" / "model" / "survivorship.json").read_text())["by_year"])
    covered = by["events"] * by["coverage"]
    missing = by["events"] - covered
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.bar(by["year"], covered, width=0.8, color=DARK, label="van árfolyam (a mintában)", edgecolor="white",
           linewidth=0.8)
    ax.bar(by["year"], missing, width=0.8, bottom=covered, color=LIGHT, hatch="////", edgecolor="white",
           linewidth=0.8, label="nincs árfolyam (kimarad)")
    for x, c, n in zip(by["year"], by["coverage"], by["events"]):
        ax.text(x, n + 25, comma(c * 100) + "%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Opportunista CFO-események száma")
    ax.set_xticks(by["year"])
    ax.set_xticklabels([str(y) if y % 2 == 0 else "" for y in by["year"]])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, by["events"].max() * 1.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "01_arfolyam_lefedettseg")


# 2. Role groups ---------------------------------------------------------------------------------------------------
ROLE_NAMES = {"cfo": "CFO", "other_officer": "Egyéb vezetők", "director": "Igazgatósági tagok",
              "ten_pct_owner": "10%-os tulajdonosok", "ceo": "CEO"}


def fig_roles() -> None:
    c = pd.read_csv(ROOT / "data" / "roles" / "cohorts.csv")
    c = c[c["strategy"] == "opportunistic"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, (period, title) in zip(axes, (("full", "2006–2026 (243 kohorsz)"), ("test", "2022–2026 (51 kohorsz)"))):
        ys = np.arange(len(ROLE_NAMES))[::-1]
        for y, g in zip(ys, ROLE_NAMES):
            nw = newey_west(c[(c["period"] == period) & (c["group"] == g)]["excess"].to_numpy(float))
            lo, hi = nw["mean"] - 1.96 * nw["se"], nw["mean"] + 1.96 * nw["se"]
            ax.plot([lo, hi], [y, y], color=DARK, linewidth=1.2)
            ax.plot(nw["mean"], y, marker="o" if g == "cfo" else "s", markersize=6.5 if g == "cfo" else 5.5,
                    color=BLACK if g == "cfo" else "white", markeredgecolor=BLACK, markeredgewidth=1.1,
                    linestyle="none")
        ax.axvline(0, color=BLACK, linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(list(ROLE_NAMES.values()))
        ax.set_title(title)
        pct_axis(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("6 havi többlethozam a SPY-hoz képest")
    save(fig, "02_szerepkorok_tobblethozam")


# 3. Event-time cumulative return by size --------------------------------------------------------------------------
def fig_event_car() -> None:
    cal = stock.market_calendar()
    prices = fe.Prices(cal)
    spy = prices(stock.MARKET_ETF)
    df, _, _ = data.load()
    size = pd.read_parquet(ROOT / "data" / "model" / "size" / "signal_size.parquet")[["event_id", "market_cap"]]
    df = df.merge(size, on="event_id", how="left")
    anchor = cal.searchsorted(df["trans_date"].to_numpy(), side="right") - 1
    entry = cal.searchsorted(df["signal_date"].to_numpy(), side="right")
    paths = np.full((len(df), EVENT_DAYS + 1), np.nan)
    for k, (sym, a) in enumerate(zip(df["symbol"], anchor)):
        if a < 0 or a + EVENT_DAYS >= len(cal):
            continue
        adj = prices(sym)
        if np.isfinite(adj[a]) and adj[a] > 0:
            paths[k] = adj[a:a + EVENT_DAYS + 1] / adj[a] - spy[a:a + EVENT_DAYS + 1] / spy[a]
    delay = int(np.median(entry - anchor))
    cap, recent = df["market_cap"], (df["signal_date"] >= "2010-01-01").to_numpy()
    groups = [("Összes (2006–2026)", np.ones(len(df), bool), BLACK, "-", None),
              ("Mikro < 300 M$", (cap < 3e8).to_numpy() & recent, DARK, "--", None),
              ("Kis 300 M$ – 2 Mrd $", ((cap >= 3e8) & (cap < 2e9)).to_numpy() & recent, MID, "-.", None),
              ("Nagy ≥ 2 Mrd $", (cap >= 2e9).to_numpy() & recent, BLACK, ":", None)]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    days = np.arange(EVENT_DAYS + 1)
    ax.axvspan(0, delay, color=PALE, linewidth=0)
    for name, m, color, ls, _ in groups:
        y = np.nanmean(paths[m], axis=0)
        ax.plot(days, y, color=color, linestyle=ls, linewidth=1.6 if ls == ":" else 1.4)
        end_label(ax, days[-1], y[-1], f"{name} (n = {int(np.isfinite(paths[m][:, -1]).sum()):,})".replace(",", " "))
    ax.axhline(0, color=BLACK, linewidth=0.6)
    ax.set_xlim(0, EVENT_DAYS)
    ax.set_xlabel("Tőzsdenapok a kereskedés napja óta")
    ax.set_ylabel("Kumulált többlethozam a SPY-hoz képest")
    pct_axis(ax, decimals=0)
    save(fig, "03_esemenyido_kumulalt_tobblethozam")


# 4. Cumulative factor alpha ---------------------------------------------------------------------------------------
def fig_cum_alpha() -> None:
    d = pd.read_parquet(ROOT / "data" / "model" / "factor_event" / "daily_returns.parquet")
    ff = sources.ff_factors_daily()
    cols = fe.MODELS["FF5+Mom"]
    series = [("Eseményidő, vásárol-és-tart, 126 nap", "event BH: all CFO buys, hold 126d", BLACK, "-"),
              ("Havi kohorszok", "cohort: all CFO buys", MID, "--")]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.axvspan(TEST_START, d.index.max(), color=PALE, linewidth=0)
    for name, col, color, ls in series:
        p = pd.concat([d[col].rename("r"), ff], axis=1, join="inner").dropna().loc["2006":]
        y = (p["r"] - p["rf"]).to_numpy()
        coef, _, _ = fe.nw_ols(y, p[cols].to_numpy(), fe.NW_DAILY)
        abnormal = pd.Series(y - p[cols].to_numpy() @ coef[1:], index=p.index)
        cum = abnormal.cumsum()
        ax.plot(cum.index, cum, color=color, linestyle=ls)
        end_label(ax, cum.index[-1], cum.iloc[-1], name)
    ax.axhline(0, color=BLACK, linewidth=0.6)
    ax.text(TEST_START + pd.Timedelta(days=40), ax.get_ylim()[0], "2022–2026", fontsize=8, color=DARK,
            va="bottom")
    ax.set_ylabel("Kumulált abnormális hozam")
    pct_axis(ax)
    save(fig, "04_kumulalt_faktoralfa")


# 5. Seed spread of configs B and C --------------------------------------------------------------------------------
def fig_seeds() -> None:
    r = pd.read_csv(ROOT / "data" / "model" / "robust" / "runs.csv")
    r = r[(r["top"] == 5) & (r["cost_bps"] == 0) & (r["seed"] != "ensemble")]
    spy = r["spy_sharpe"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    for ax, (col, label, ref, is_pct) in zip(axes, (("cagr", "CAGR", 0.134, True),
                                                    ("sharpe", "Sharpe-mutató", spy, False))):
        for x, cfg in enumerate(("B", "C")):
            v = r[r["config"] == cfg]
            jitter = np.linspace(-0.12, 0.12, len(v))
            main = v["seed"].astype(str) == "42"
            ax.plot(x + jitter[~main.to_numpy()], v.loc[~main, col], "o", markersize=5, color="white",
                    markeredgecolor=DARK, markeredgewidth=1)
            ax.plot(x + jitter[main.to_numpy()], v.loc[main, col], "o", markersize=6, color=BLACK)
            med = v[col].median()
            ax.plot([x - 0.25, x + 0.25], [med, med], color=BLACK, linewidth=2)
        ax.axhline(ref, color=MID, linestyle="--", linewidth=1)
        ax.text(1.58, ref, "SPY", ha="right", va="bottom", fontsize=8, color=DARK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["B konfiguráció", "C konfiguráció"])
        ax.set_xlim(-0.5, 1.6)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
        (pct_axis(ax) if is_pct else num_axis(ax))
    save(fig, "05_seedek_szorasa")


# 6. Rolling IC by year --------------------------------------------------------------------------------------------
def fig_rolling_ic() -> None:
    rows = {}
    for s in (42, 1, 2, 3, 4):
        for y in range(2012, 2027):
            rep = json.loads((ROOT / "data" / "model" / "rolling" / f"seed{s}" / str(y) / "report.json").read_text())
            rows[(s, y)] = rep["test"]["ic_within_month_mean"]
    ic = pd.Series(rows).unstack(0)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    colors = [LIGHT if y < 2022 else DARK for y in ic.index]
    ax.bar(ic.index, ic.mean(axis=1), width=0.7, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(ic.index, ic.mean(axis=1), yerr=[ic.mean(axis=1) - ic.min(axis=1), ic.max(axis=1) - ic.mean(axis=1)],
                fmt="none", ecolor=BLACK, elinewidth=0.8, capsize=2)
    ax.axhline(0, color=BLACK, linewidth=0.6)
    means = []
    for a, b in ((2012, 2021), (2022, 2026)):
        m = ic.loc[a:b].mean(axis=1).mean()
        ax.plot([a - 0.4, b + 0.4], [m, m], color=BLACK, linestyle="--", linewidth=0.9)
        means.append(f"{a}–{b}: {comma(m, 2)}")
    ax.text(2011.6, ic.max().max(), "Szaggatott vonal: időszaki átlag\n" + "\n".join(means), fontsize=8,
            va="top", ha="left")
    ax.set_xticks(ic.index)
    ax.set_xticklabels([str(y) for y in ic.index], rotation=45)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Hónapon belüli Spearman IC")
    num_axis(ax, decimals=2)
    save(fig, "06_gordulo_ic_evente")


# 7. Rolling model portfolio against benchmarks --------------------------------------------------------------------
def fig_rolling_wealth() -> None:
    d = pd.read_parquet(ROOT / "data" / "model" / "factor_event" / "daily_returns.parquet")
    series = [("Modell legjobb öt (együttes)", "cohort: model top 5 (ensemble)", BLACK, "-"),
              ("Összes CFO-vétel", "cohort: all CFO buys (2012-)", MID, "--"),
              ("5 legalacsonyabb volatilitású", "cohort: lowest volatility 5 (2012-)", DARK, ":"),
              ("SPY", "SPY", BLACK, "-.")]
    start = d["cohort: model top 5 (ensemble)"].dropna().index.min()
    end = d["cohort: model top 5 (ensemble)"].dropna().index.max()
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.axvspan(TEST_START, end, color=PALE, linewidth=0)
    finals = []
    for name, col, color, ls in series:
        w = (1 + d.loc[start:end, col].fillna(0)).cumprod()
        ax.plot(w.index, w, color=color, linestyle=ls, linewidth=1.7 if ls == ":" else 1.4)
        finals.append((w.iloc[-1], name))
    ax.set_yscale("log")
    ax.set_yticks([0.75, 1, 1.5, 2, 3, 4, 6, 8])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v, 2 if v < 1 else (1 if v % 1 else 0))))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    # labels at the line ends, pushed apart to at least MIN_GAP in log10 units
    min_gap, placed = 0.075, []
    for v, name in sorted(finals):
        y = np.log10(v) if not placed else max(np.log10(v), placed[-1] + min_gap)
        placed.append(y)
        ax.annotate(f"{name}: {comma(v, 1)}×", (end, v), xytext=(end + pd.Timedelta(days=60), 10 ** y),
                    va="center", fontsize=8.5, annotation_clip=False)
    ax.set_ylabel("1 dollár értéke (log skála)")
    save(fig, "07_gordulo_portfoliok")


# 8. Alpha by size and period --------------------------------------------------------------------------------------
def fig_alpha_by_size() -> None:
    a = pd.read_csv(ROOT / "data" / "model" / "factor_event" / "alphas.csv")
    a = a[a["model"] == "FF5+Mom"]
    groups = [("Mikro\n< 300 M$", "event BH: micro < 300m, hold 126d"),
              ("Kis\n300 M$ – 2 Mrd $", "event BH: small 300m-2bn, hold 126d"),
              ("Nagy\n≥ 2 Mrd $", "event BH: cap >= 2bn, hold 126d"),
              ("Összes", "event BH: all CFO buys, hold 126d")]
    periods = [("2006-2026", "teljes időszak*", DARK, None), ("2012-2021", "2012–2021", LIGHT, None),
               ("2022-2026", "2022–2026", "white", "////")]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    w = 0.26
    for k, (per, label, color, hatch) in enumerate(periods):
        for x, (_, port) in enumerate(groups):
            r = a[(a["portfolio"] == port) & (a["period"] == per)].iloc[0]
            se = abs(r["alpha_ann"] / r["t_alpha"])
            ax.bar(x + (k - 1) * w, r["alpha_ann"], width=w, color=color, hatch=hatch, edgecolor=BLACK,
                   linewidth=0.6, label=label if x == 0 else None)
            ax.errorbar(x + (k - 1) * w, r["alpha_ann"], yerr=1.96 * se, fmt="none", ecolor=BLACK, elinewidth=0.8,
                        capsize=2)
    ax.axhline(0, color=BLACK, linewidth=0.6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Éves FF5 + momentum alfa")
    pct_axis(ax)
    ax.legend(loc="upper right", ncol=3)
    save(fig, "08_alfa_meret_szerint")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for f in (fig_coverage, fig_roles, fig_event_car, fig_cum_alpha, fig_seeds, fig_rolling_ic, fig_rolling_wealth,
              fig_alpha_by_size):
        f()
    return 0


if __name__ == "__main__":
    sys.exit(main())
