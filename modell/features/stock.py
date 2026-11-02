# Price-based features: Yahoo symbol resolution, stock returns, risk, liquidity, sector and market returns.

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

import sources

MARKET_ETF = "SPY"
# Yahoo sector -> sector ETF. XLRE (2015) and XLC (2018) start too late for the sample, so IYR and VOX stand in.
SECTOR_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Industrials": "XLI", "Energy": "XLE", "Utilities": "XLU",
    "Basic Materials": "XLB", "Real Estate": "IYR", "Communication Services": "VOX",
}
RET_HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "6m": 126, "12m": 252}  # market days
SEC_HORIZONS = ("3m", "6m", "12m")
FFILL_LIMIT = 10  # a price older than 10 market days is not used as current
MATCH_TOL = 1.25
ANN = np.sqrt(252)

STOCK_FEATURES = (
    [f"stk_ret_{k}" for k in RET_HORIZONS]
    + ["stk_vol_1m", "stk_vol_3m", "stk_vol_12m", "stk_downside_vol_3m", "stk_beta_12m"]
    + ["stk_dist_52w_high", "stk_dist_52w_low", "stk_log_price", "stk_log_adv_1m", "stk_volume_ratio",
       "stk_days_since_price"]
)
SECTOR_FEATURES = (["sector"] + [f"sec_ret_{h}" for h in SEC_HORIZONS] + [f"mkt_ret_{h}" for h in SEC_HORIZONS]
                   + [f"rel_sec_ret_{h}" for h in SEC_HORIZONS])

_PREFIX = re.compile(r"\b(?:NYSE ?AMERICAN|NYSE ?MKT|NYSE|NASDAQ|NASD|AMEX|OTCBB|OTCQB|OTCQX|OTC|PINK|ARCA|TSX)\s*[:\-]\s*")
_OTC_SUFFIX = re.compile(r"[.\-](?:OB|PK|BB|OTC|QB|QX|PINK)$")
_JUNK = {"NONE", "NA", "N", "NULL", "UNASSIGNED", "UNKNOWN", "TBD", "OTC", "NYSE", "NASDAQ", "AMEX"}

log = logging.getLogger("features.stock")


def clean_ticker(raw) -> list[str]:
    """Yahoo symbols in a filed ticker: '(NYSE:FBC)' -> FBC; 'GEF,GEF.B' -> GEF, GEF-B; 'DNBF.OB' -> DNBF."""
    if not isinstance(raw, str):
        return []
    s = _PREFIX.sub(" ", raw.upper())
    s = re.sub(r"[()$\"']", " ", s)
    out = []
    for tok in re.split(r"[,;&|\s]+|\bAND\b", s):
        tok = _OTC_SUFFIX.sub("", tok.strip(".-"))
        tok = re.sub(r"[./](?=[A-Z]$)", "-", tok)  # class shares: BRK.B, BRK/B -> BRK-B
        if tok and tok not in _JUNK and re.fullmatch(r"[A-Z0-9]{1,6}(?:-[A-Z])?", tok):
            out.append(tok)
    return list(dict.fromkeys(out))


def candidate_symbols(events: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """(issuer_cik, symbol, source) with source 'filed', 'sec' or 'filed+sec'."""
    filed = events[["issuer_cik", "issuer_ticker"]].drop_duplicates()
    rows = [(cik, sym, "filed") for cik, t in filed.itertuples(index=False) for sym in clean_ticker(t)]
    own = sec[sec["cik"].isin(events["issuer_cik"])]
    rows += [(cik, t.upper().replace(".", "-"), "sec") for cik, t in own[["cik", "ticker"]].itertuples(index=False)]
    c = pd.DataFrame(rows, columns=["issuer_cik", "symbol", "source"])
    return c.groupby(["issuer_cik", "symbol"], as_index=False)["source"].agg(lambda s: "+".join(sorted(set(s))))


def market_calendar() -> pd.DatetimeIndex:
    return sources.load_prices(MARKET_ETF).index


def anchor_dates(dates: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    """Last market day on or before each date (NaT before the calendar starts)."""
    pos = cal.searchsorted(dates.to_numpy(), side="right") - 1
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = pos >= 0
    out[ok] = cal[pos[ok]]
    return out


def prepare(px: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Prices on the market calendar plus raw_close, the price as it traded (Yahoo's close is split-adjusted)."""
    px = px[~px.index.duplicated()].sort_index()
    split = px["split"].fillna(0).replace(0, 1.0) if "split" in px else pd.Series(1.0, index=px.index)
    later_splits = split[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    return px.assign(raw_close=px["close"] * later_splits).reindex(cal)


def stock_daily(p: pd.DataFrame, mkt_lr: pd.Series) -> pd.DataFrame:
    """Daily values of the stock features (and helper columns) on the market calendar."""
    adj = p["adj_close"].ffill(limit=FFILL_LIMIT)
    close = p["close"].ffill(limit=FFILL_LIMIT)
    lr = np.log(p["adj_close"]).diff()
    n_3m = lr.rolling(63).count()
    f = {f"stk_ret_{k}": adj / adj.shift(h) - 1 for k, h in RET_HORIZONS.items()}
    f["stk_vol_1m"] = lr.rolling(21, min_periods=15).std() * ANN
    f["stk_vol_3m"] = lr.rolling(63, min_periods=40).std() * ANN
    f["stk_vol_12m"] = lr.rolling(252, min_periods=160).std() * ANN
    # root mean square of the down-day returns only
    down = lr.where(lr < 0)
    f["stk_downside_vol_3m"] = (np.sqrt((down ** 2).rolling(63, min_periods=5).mean()) * ANN).where(n_3m >= 40)
    m = mkt_lr.where(lr.notna())
    f["stk_beta_12m"] = lr.rolling(252, min_periods=120).cov(m) / m.rolling(252, min_periods=120).var()
    f["stk_dist_52w_high"] = close / p["high"].rolling(252, min_periods=120).max() - 1
    f["stk_dist_52w_low"] = close / p["low"].rolling(252, min_periods=120).min() - 1
    raw = p["raw_close"].ffill(limit=FFILL_LIMIT)
    f["stk_log_price"] = np.log(raw)
    # split-adjusted close x split-adjusted volume = dollar volume as traded
    adv = (p["close"] * p["volume"]).rolling(21, min_periods=10).mean()
    f["stk_log_adv_1m"] = np.log1p(adv)
    vol_3m = p["volume"].rolling(63, min_periods=40).mean()
    f["stk_volume_ratio"] = p["volume"].rolling(5, min_periods=3).mean() / vol_3m.where(vol_3m > 0)
    f["adv_1m"] = adv
    f["raw_close"] = raw
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)


def resolve_symbols(events: pd.DataFrame, cands: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per event: symbol, symbol_status (confirmed / issuer / none), price_ratio (vwap / close)."""
    pairs = events[["event_id", "issuer_cik", "anchor", "vwap"]].merge(cands, on="issuer_cik")
    pairs["raw_close"] = np.nan
    for sym, idx in pairs.groupby("symbol").groups.items():
        px = sources.load_prices(sym)
        if px is None:
            continue
        rc = prepare(px, cal)["raw_close"].ffill(limit=FFILL_LIMIT)
        pairs.loc[idx, "raw_close"] = rc.reindex(pairs.loc[idx, "anchor"]).to_numpy()
    pairs["abs_lr"] = np.log(pairs["vwap"] / pairs["raw_close"]).abs()
    pairs["match"] = pairs["abs_lr"] <= np.log(MATCH_TOL)
    score = pairs.groupby(["issuer_cik", "symbol"]).agg(n_match=("match", "sum"),
                                                       n_checked=("abs_lr", "count"))
    pairs = pairs.join(score, on=["issuer_cik", "symbol"])
    pairs["sec"] = pairs["source"].str.contains("sec")

    confirmed = (pairs[pairs["match"]]
                 .sort_values(["event_id", "abs_lr", "n_match", "sec"], ascending=[True, True, False, False])
                 .drop_duplicates("event_id").assign(symbol_status="confirmed"))
    rest = pairs[~pairs["event_id"].isin(confirmed["event_id"]) & pairs["raw_close"].notna()
                 & (pairs["n_match"] >= 1) & (pairs["n_match"] >= 0.5 * pairs["n_checked"])]
    fallback = (rest.sort_values(["event_id", "n_match", "sec"], ascending=[True, False, False])
                .drop_duplicates("event_id").assign(symbol_status="issuer"))
    # the SEC's ticker is mapped by CIK, so it is the right company even when no price confirms it
    done = set(confirmed["event_id"]) | set(fallback["event_id"])
    sec_only = (pairs[~pairs["event_id"].isin(done) & pairs["sec"] & pairs["raw_close"].notna()]
                .sort_values(["event_id", "abs_lr"]).drop_duplicates("event_id").assign(symbol_status="sec_unconfirmed"))
    chosen = pd.concat([confirmed, fallback, sec_only])
    out = events[["event_id"]].merge(chosen[["event_id", "symbol", "source", "symbol_status", "raw_close"]],
                                     on="event_id", how="left")
    out["symbol_status"] = out["symbol_status"].fillna("none")
    has_cand = events["issuer_cik"].isin(cands["issuer_cik"]).to_numpy()
    has_data = events["event_id"].isin(pairs.loc[pairs["raw_close"].notna(), "event_id"]).to_numpy()
    out.loc[(out["symbol_status"] == "none") & ~has_cand, "symbol_status"] = "none_no_ticker"
    out.loc[(out["symbol_status"] == "none") & has_cand & ~has_data, "symbol_status"] = "none_no_yahoo_data"
    out.loc[out["symbol_status"] == "none", "symbol_status"] = "none_price_mismatch"
    return out.rename(columns={"source": "symbol_source"}).drop(columns="raw_close")


def etf_returns(symbol: str, cal: pd.DatetimeIndex, horizons=SEC_HORIZONS) -> pd.DataFrame:
    px = sources.load_prices(symbol)
    if px is None:
        return pd.DataFrame(index=cal, columns=list(horizons), dtype=float)
    adj = prepare(px, cal)["adj_close"].ffill(limit=FFILL_LIMIT)
    return pd.DataFrame({h: adj / adj.shift(RET_HORIZONS[h]) - 1 for h in horizons})


def price_features(events: pd.DataFrame, choice: pd.DataFrame, sectors: dict[str, str | None],
                   cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Stock, sector, market and price-dependent trade features, indexed by event_id."""
    spy = prepare(sources.load_prices(MARKET_ETF), cal)
    mkt_lr = np.log(spy["adj_close"]).diff()
    ev = events[["event_id", "trans_date", "anchor", "vwap", "value_usd"]].merge(choice, on="event_id")
    cols = STOCK_FEATURES + ["adv_1m", "raw_close"]
    out = pd.DataFrame(np.nan, index=ev["event_id"], columns=cols)

    matched = ev[ev["symbol"].notna()]
    for n, (sym, g) in enumerate(matched.groupby("symbol")):
        p = prepare(sources.load_prices(sym), cal)
        f = stock_daily(p, mkt_lr)
        vals = f.reindex(g["anchor"])
        out.loc[g["event_id"], vals.columns] = vals.to_numpy()
        traded = p.index[p["volume"].fillna(0) > 0]
        if len(traded):
            pos = traded.searchsorted(g["trans_date"].to_numpy(), side="right") - 1
            last = np.where(pos >= 0, traded[np.clip(pos, 0, None)].to_numpy(), np.datetime64("NaT"))
            out.loc[g["event_id"], "stk_days_since_price"] = ((g["trans_date"].to_numpy() - last)
                                                              / np.timedelta64(1, "D"))
        if n % 500 == 0:
            log.info("price features: %d / %d symbols", n, matched["symbol"].nunique())
    # a stale last trade means the anchor values are stale too; days_since_price shows it

    out["sector"] = ev.set_index("event_id")["symbol"].map(sectors)
    mkt = etf_returns(MARKET_ETF, cal)
    anchors = ev.set_index("event_id")["anchor"]
    for h in SEC_HORIZONS:
        out[f"mkt_ret_{h}"] = mkt[h].reindex(anchors).to_numpy()
        out[f"sec_ret_{h}"] = np.nan
    for sector, etf in SECTOR_ETF.items():
        ids = out.index[out["sector"] == sector]
        r = etf_returns(etf, cal)
        for h in SEC_HORIZONS:
            out.loc[ids, f"sec_ret_{h}"] = r[h].reindex(anchors[ids]).to_numpy()
    for h in SEC_HORIZONS:
        out[f"rel_sec_ret_{h}"] = out[f"stk_ret_{h}"] - out[f"sec_ret_{h}"]

    trade = ev.set_index("event_id")
    out["price_vs_close"] = trade["vwap"] / out["raw_close"] - 1
    out["trade_value_to_adv"] = trade["value_usd"] / out["adv_1m"].where(out["adv_1m"] > 0)
    return out.drop(columns=["adv_1m", "raw_close"])
