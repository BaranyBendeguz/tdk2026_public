# Download the SEC Insider Transactions Data Sets (Forms 3, 4, 5).

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
SEC_BASE = "https://www.sec.gov"
USER_AGENT = "Your Name your@email.com"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"
README_PDF = RAW_DIR / "insider_transactions_readme.pdf"

# SEC fair access limit is 10 requests/second; stay well below it.
MIN_INTERVAL_S = 0.2
MAX_RETRIES = 6
RETRY_STATUS = {403, 429, 500, 502, 503, 504}

ZIP_RE = re.compile(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', re.IGNORECASE)
README_RE = re.compile(r'href="([^"]*insider_transactions_readme\.pdf)"', re.IGNORECASE)

log = logging.getLogger("download")


class Client:
    """requests.Session wrapper with throttling and retry with backoff."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}
        )
        self._last = 0.0

    def _throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str, stream: bool = False) -> requests.Response:
        delay = 2.0
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self.session.get(url, stream=stream, timeout=120)
            except requests.RequestException as exc:
                log.warning("GET %s failed (%s), attempt %d/%d", url, exc, attempt, MAX_RETRIES)
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code not in RETRY_STATUS:
                    resp.raise_for_status()
                log.warning("GET %s -> HTTP %d, attempt %d/%d", url, resp.status_code, attempt, MAX_RETRIES)
                resp.close()
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts")

    def download(self, url: str, dest: Path) -> int:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self.get(url, stream=True) as resp:
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
        return dest.stat().st_size


def discover(client: Client) -> tuple[dict[str, str], str | None]:
    """Return ({quarter: absolute zip url}, readme url) scraped from the index page."""
    html = client.get(INDEX_URL).text
    (RAW_DIR / "index.html").write_text(html, encoding="utf-8")
    quarters: dict[str, str] = {}
    for href, quarter in ZIP_RE.findall(html):
        quarters[quarter.lower()] = urljoin(SEC_BASE, href)
    m = README_RE.search(html)
    readme = urljoin(SEC_BASE, m.group(1)) if m else None
    return dict(sorted(quarters.items())), readme


def expected_quarters(first: str, last: str) -> list[str]:
    y, q = int(first[:4]), int(first[5])
    ly, lq = int(last[:4]), int(last[5])
    out = []
    while (y, q) <= (ly, lq):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="download only these quarters, e.g. 2025q3")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = Client()

    quarters, readme_url = discover(client)
    if not quarters:
        log.error("no quarterly zip links found on %s", INDEX_URL)
        return 1
    found = list(quarters)
    log.info("found %d quarters on index page: %s .. %s", len(found), found[0], found[-1])
    missing = sorted(set(expected_quarters(found[0], found[-1])) - set(found))
    if missing:
        log.warning("quarters missing from index page: %s", ", ".join(missing))

    if readme_url:
        if README_PDF.exists():
            log.info("readme cached: %s", README_PDF.name)
        else:
            size = client.download(readme_url, README_PDF)
            log.info("downloaded readme %s (%d bytes)", readme_url, size)
    else:
        log.warning("readme link not found on index page")

    todo = {q: u for q, u in quarters.items() if not args.only or q in set(args.only)}
    failed: list[str] = []
    for quarter, url in todo.items():
        dest = RAW_DIR / f"{quarter}_form345.zip"
        if dest.exists() and dest.stat().st_size > 0:
            log.info("%s cached (%d bytes)", quarter, dest.stat().st_size)
            continue
        try:
            size = client.download(url, dest)
            log.info("%s downloaded (%d bytes) from %s", quarter, size, url)
        except Exception as exc:  # keep going, report at the end
            log.error("%s FAILED: %s", quarter, exc)
            failed.append(quarter)

    MANIFEST.write_text(
        json.dumps(
            {
                "index_url": INDEX_URL,
                "readme_url": readme_url,
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "quarters": quarters,
                "missing_from_index": missing,
                "failed": failed,
            },
            indent=2,
        )
    )
    log.info("quarters found: %s", ", ".join(found))
    if failed:
        log.error("failed quarters: %s", ", ".join(failed))
        return 2
    log.info("all %d quarters present in %s", len(todo), RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
