# Downloads and local caches for the feature build.

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "data" / "market"
PRICE_DIR = MARKET / "prices"
NO_DATA = MARKET / "yahoo_no_data.json"
INFO_CACHE = MARKET / "yahoo_info.json"
SEC_TICKERS = MARKET / "sec_company_tickers.json"
FRED_DIR = MARKET / "fred"
ALFRED_DIR = MARKET / "alfred"
SEC_SHARES_DIR = MARKET / "sec_shares"
SP500_FILE = MARKET / "sp500_components.csv"
FF_DIR = MARKET / "ff"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FILES = ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip")
SP500_COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
SP500_URL = (f"https://raw.githubusercontent.com/fja05680/sp500/{SP500_COMMIT}/"
             "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")

PRICE_START = "1998-01-01"
SEC_USER_AGENT = "Your Name your@email.com"  # same as src/download.py
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALFRED_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
PRICE_COLS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
              "Volume": "volume", "Stock Splits": "split"}
# yfinance error texts that mean "Yahoo has no such symbol", as opposed to a transient failure
NO_DATA_ERRORS = ("delisted", "not found", "no data found", "no price data", "no timezone")

log = logging.getLogger("features.sources")


def price_path(symbol: str) -> Path:
    return PRICE_DIR / f"{symbol}.parquet"


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def yahoo_prices(symbols, batch_size: int = 40, pause: float = 1.0, refresh: bool = False) -> None:
    """Download missing daily price histories into data/market/prices/{symbol}.parquet.

    Symbols Yahoo does not know are remembered in yahoo_no_data.json and not requested again (unless refresh).
    Transient errors (rate limit, timeouts) leave the symbol for the next run.
    """
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    no_data = set() if refresh else set(_load_json(NO_DATA, []))
    todo = sorted(s for s in set(symbols) if s not in no_data and (refresh or not price_path(s).exists()))
    log.info("yahoo prices: %d symbols to download (%d cached, %d known missing)",
             len(todo), len(set(symbols)) - len(todo) - len(no_data & set(symbols)), len(no_data & set(symbols)))
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        for attempt in range(5):
            data = yf.download(chunk, start=PRICE_START, auto_adjust=False, actions=True, group_by="ticker",
                               threads=True, progress=False)
            errors = {k: str(v).lower() for k, v in yf.shared._ERRORS.items()}
            transient = [s for s, e in errors.items() if not any(m in e for m in NO_DATA_ERRORS)]
            if not transient or attempt == 4:
                break
            wait = 30 * (attempt + 1)
            log.warning("yahoo: %d transient errors (%s), retrying in %ds", len(transient),
                        errors[transient[0]][:80], wait)
            time.sleep(wait)
        for s in chunk:
            x = data[s] if isinstance(data.columns, pd.MultiIndex) and s in data.columns.get_level_values(0) else None
            if x is not None:
                x = x.rename(columns=PRICE_COLS)
                # yfinance occasionally returns a column as object dtype; parquet needs numbers
                x = x[[c for c in PRICE_COLS.values() if c in x.columns]].apply(pd.to_numeric, errors="coerce")
                if "close" not in x.columns:
                    log.warning("yahoo %s: unexpected columns %s", s, list(x.columns))
                    continue
                x = x.dropna(subset=["close"])
            if x is not None and len(x):
                x.index = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
                x.index.name = "date"
                x.to_parquet(price_path(s))
            elif any(m in errors.get(s, "no data found") for m in NO_DATA_ERRORS):
                no_data.add(s)
        NO_DATA.write_text(json.dumps(sorted(no_data)))
        if (i // batch_size) % 10 == 0:
            log.info("yahoo prices: %d / %d", min(i + batch_size, len(todo)), len(todo))
        time.sleep(pause)


def load_prices(symbol: str) -> pd.DataFrame | None:
    p = price_path(symbol)
    return pd.read_parquet(p) if p.exists() else None


def yahoo_info(symbols, workers: int = 4) -> dict[str, dict]:
    """Yahoo sector / industry / quote type per symbol, cached in yahoo_info.json."""
    cache = _load_json(INFO_CACHE, {})
    todo = sorted(s for s in set(symbols) if s not in cache)
    log.info("yahoo info: %d symbols to fetch (%d cached)", len(todo), len(set(symbols)) - len(todo))

    def fetch(s: str) -> tuple[str, dict | None]:
        for attempt in range(4):
            try:
                i = yf.Ticker(s).info or {}
                return s, {k: i.get(k) for k in ("sector", "industry", "quoteType", "longName", "exchange")}
            except Exception as e:  # noqa: BLE001 - yfinance raises many types
                msg = str(e).lower()
                if "404" in msg or "not found" in msg:
                    return s, {"error": "not found"}
                time.sleep(10 * (attempt + 1))
        return s, None

    for i in range(0, len(todo), 200):
        with ThreadPoolExecutor(workers) as ex:
            for s, info in ex.map(fetch, todo[i:i + 200]):
                if info is not None:
                    cache[s] = info
        INFO_CACHE.write_text(json.dumps(cache))
        log.info("yahoo info: %d / %d", min(i + 200, len(todo)), len(todo))
    return {s: cache[s] for s in symbols if s in cache}


def sec_company_tickers(refresh: bool = False) -> pd.DataFrame:
    """Current CIK -> ticker map published by the SEC (one row per ticker)."""
    if refresh or not SEC_TICKERS.exists():
        MARKET.mkdir(parents=True, exist_ok=True)
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
        r.raise_for_status()
        SEC_TICKERS.write_bytes(r.content)
    rows = json.loads(SEC_TICKERS.read_text()).values()
    return pd.DataFrame(rows).rename(columns={"cik_str": "cik"})[["cik", "ticker", "title"]]


def sec_shares_outstanding(ciks, pause: float = 0.2) -> pd.DataFrame:
    """Shares outstanding reported in XBRL filings (SEC companyfacts API), one row per reported value.

    Concepts: dei:EntityCommonStockSharesOutstanding (cover page, as of a date close to the filing) and
    us-gaap:CommonStockSharesOutstanding (balance sheet date). Each value keeps its `end` (as-of) and `filed` dates, so
    a caller can use only what was public on a given day. Cached per issuer in data/market/sec_shares/{cik}.parquet;
    issuers without XBRL data get an empty file and are not requested again.
    """
    SEC_SHARES_DIR.mkdir(parents=True, exist_ok=True)
    concepts = (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"))
    cols = ["cik", "concept", "end", "filed", "form", "val"]
    todo = sorted({int(c) for c in ciks} - {int(p.stem) for p in SEC_SHARES_DIR.glob("*.parquet")})
    log.info("sec shares: %d issuers to download", len(todo))
    for i, cik in enumerate(todo):
        for attempt in range(5):
            r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                             headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
            if r.status_code in (200, 404):
                break
            time.sleep(5 * (attempt + 1))
        else:
            r.raise_for_status()
        rows = []
        if r.status_code == 200:
            facts = r.json().get("facts", {})
            for ns, name in concepts:
                for v in facts.get(ns, {}).get(name, {}).get("units", {}).get("shares", []):
                    rows.append({"cik": cik, "concept": name, "end": v.get("end"), "filed": v.get("filed"),
                                 "form": v.get("form"), "val": v.get("val")})
        df = pd.DataFrame(rows, columns=cols).astype({"cik": "int64", "val": "float64"})
        df[["end", "filed"]] = df[["end", "filed"]].apply(pd.to_datetime)
        df.to_parquet(SEC_SHARES_DIR / f"{cik}.parquet", index=False)
        if i % 200 == 199:
            log.info("sec shares: %d / %d", i + 1, len(todo))
        time.sleep(pause)
    parts = [pd.read_parquet(SEC_SHARES_DIR / f"{int(c)}.parquet") for c in sorted({int(c) for c in ciks})]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def sp500_components(refresh: bool = False) -> pd.DataFrame:
    """Historical S&P 500 membership, one row per (date, ticker) in long format: the rows of the data set are the
    days on which the composition changed, each with the full list of tickers valid from that day.

    Source: github.com/fja05680/sp500 (MIT license), pinned to commit SP500_COMMIT; built from the list of
    A. Clenow (Trading Evolved) to 2019 and the S&P 500 Wikipedia changes since. Tickers are those used on the day.
    """
    if refresh or not SP500_FILE.exists():
        MARKET.mkdir(parents=True, exist_ok=True)
        r = requests.get(SP500_URL, timeout=120)
        r.raise_for_status()
        SP500_FILE.write_bytes(r.content)
    df = pd.read_csv(SP500_FILE, parse_dates=["date"])
    df["ticker"] = df["tickers"].str.split(",")
    return df.drop(columns="tickers").explode("ticker").reset_index(drop=True)


def ff_factors_daily(refresh: bool = False) -> pd.DataFrame:
    """Daily Fama-French 5 factors (2x3) and momentum from Kenneth French's data library, as decimals.

    Columns: mkt_rf, smb, hml, rmw, cma, rf, mom. Cached in data/market/ff/ (the zip files as downloaded).
    """
    FF_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in FF_FILES:
        path = FF_DIR / name
        if refresh or not path.exists():
            r = requests.get(FF_URL + name, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        lines = text.splitlines()
        header = next(i for i, ln in enumerate(lines) if ln.startswith(","))
        rows = [ln for ln in lines[header + 1:] if ln[:8].strip().isdigit() and len(ln.split(",")[0].strip()) == 8]
        df = pd.read_csv(io.StringIO("\n".join(rows)), header=None,
                         names=["date"] + [c.strip() for c in lines[header].split(",")[1:]])
        df.index = pd.to_datetime(df.pop("date").astype(str), format="%Y%m%d")
        frames.append(df.apply(pd.to_numeric, errors="coerce") / 100)
    out = pd.concat(frames, axis=1, join="inner")
    return out.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma",
                               "RF": "rf", "Mom": "mom"})


def _get(url: str, params: dict, tries: int = 5) -> str:
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.text
            log.warning("%s: HTTP %d", url, r.status_code)
        except requests.RequestException as e:
            log.warning("%s: %s", url, e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url} {params}")


def fred_daily(series: list[str], refresh: bool = False) -> pd.DataFrame:
    """Current FRED values of daily series, one column per series (cached in data/market/fred/{series}.csv)."""
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    cols = []
    for s in series:
        path = FRED_DIR / f"{s}.csv"
        if refresh or not path.exists():
            path.write_text(_get(FRED_URL, {"id": s, "cosd": "1997-01-01"}))
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        cols.append(pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(s))
    return pd.concat(cols, axis=1).sort_index()


def alfred_vintages(series: str, vintage_dates, n_last: int = 13, per_request: int = 12) -> pd.DataFrame:
    """The last n_last observations of `series` as published on each vintage date.

    ALFRED returns at most 12 series per graph, so 12 vintages go in one request. Returns long format
    (vintage_date, obs_date, value), cached in data/market/alfred/{series}.parquet.
    """
    path = ALFRED_DIR / f"{series}.parquet"
    cache = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["vintage_date", "obs_date", "value"])
    want = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(vintage_dates)))))
    todo = want.difference(pd.DatetimeIndex(cache["vintage_date"].unique()))
    log.info("alfred %s: %d vintages to fetch (%d cached)", series, len(todo), len(want) - len(todo))
    parts = [cache]
    for i in range(0, len(todo), per_request):
        chunk = todo[i:i + per_request]
        text = _get(ALFRED_URL, {"id": ",".join([series] * len(chunk)),
                                 "vintage_date": ",".join(d.strftime("%Y-%m-%d") for d in chunk)})
        df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
        if len(df.columns) != len(chunk):
            raise RuntimeError(f"alfred {series}: asked for {len(chunk)} vintages, got {len(df.columns)}")
        for col in df.columns:
            v = df[col].dropna().tail(n_last)
            vintage = pd.Timestamp(col.rsplit("_", 1)[1])
            if v.empty:  # no vintage that early: keep a marker row so it is not requested again
                v = pd.Series([float("nan")], index=pd.DatetimeIndex([pd.NaT]))
            parts.append(pd.DataFrame({"vintage_date": vintage, "obs_date": v.index, "value": v.to_numpy()}))
        if (i // per_request) % 50 == 49 or i + per_request >= len(todo):
            log.info("alfred %s: %d / %d", series, i + len(chunk), len(todo))
            ALFRED_DIR.mkdir(parents=True, exist_ok=True)
            pd.concat([p for p in parts if len(p)], ignore_index=True).to_parquet(path, index=False)
        time.sleep(0.5)
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return out[out["vintage_date"].isin(want)]
