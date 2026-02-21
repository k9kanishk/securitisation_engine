from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

import requests


def cik10(cik: str | int) -> str:
    s = str(cik).strip()
    if not s.isdigit():
        raise ValueError(f"CIK must be digits only, got: {cik}")
    return s.zfill(10)


@dataclass(frozen=True)
class FilingRef:
    cik: str                 # 10-digit
    accession_no: str        # no dashes
    filing_date: str         # YYYY-MM-DD (as provided)
    form: str
    primary_document: str

    @property
    def base_dir_url(self) -> str:
        # /Archives/edgar/data/{cik_without_leading_zeros}/{accession_no}/
        cik_int = str(int(self.cik))
        return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{self.accession_no}/"

    @property
    def primary_doc_url(self) -> str:
        return urljoin(self.base_dir_url, self.primary_document)


class SECEdgarClient:
    """
    Minimal SEC EDGAR client for:
      - data.sec.gov submissions JSON
      - sec.gov Archives HTML filings and exhibits

    IMPORTANT: Provide a real User-Agent string (name + email) to comply with SEC fair access guidance.
    """
    def __init__(self, user_agent: str, min_interval_s: float = 0.25, timeout_s: int = 30):
        ua = (user_agent or "").strip()
        if not ua:
            raise ValueError(
                "SEC User-Agent required. Example: 'YourName your.email@domain.com'"
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "text/html,application/json",
            }
        )
        self.min_interval_s = float(min_interval_s)
        self.timeout_s = int(timeout_s)
        self._last_request_ts = 0.0

    def _rate_limit(self) -> None:
        now = time.time()
        wait = self.min_interval_s - (now - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.time()

    def get_text(self, url: str) -> str:
        self._rate_limit()
        r = self.session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return r.text

    def get_json(self, url: str) -> Dict[str, Any]:
        self._rate_limit()
        r = self.session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def get_submissions(self, cik: str | int) -> Dict[str, Any]:
        """
        SEC docs: data.sec.gov/submissions/CIK##########.json
        """
        c10 = cik10(cik)
        url = f"https://data.sec.gov/submissions/CIK{c10}.json"
        return self.get_json(url)

    def latest_filing(self, cik: str | int, form: str = "10-D") -> FilingRef:
        sub = self.get_submissions(cik)
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accs = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        prim_docs = recent.get("primaryDocument", [])

        for f, acc, dt, prim in zip(forms, accs, dates, prim_docs):
            if str(f).strip().upper() == form.upper():
                acc_nodash = str(acc).replace("-", "")
                return FilingRef(
                    cik=cik10(cik),
                    accession_no=acc_nodash,
                    filing_date=str(dt),
                    form=str(f),
                    primary_document=str(prim),
                )

        raise ValueError(f"No {form} found in recent filings for CIK {cik10(cik)}")

    def find_first_exhibit_url(self, filing_primary_html: str, filing_base_dir_url: str) -> str:
        """
        Finds first Exhibit 99.1-ish link inside the 10-D primary HTML.
        BMW uses names like: bmw2025-a_exhibit991.htm
        """
        import re

        # common patterns we see in BMW 10-Ds
        patterns = [
            r'href="([^"]+exhibit99[^"]*\.htm)"',
            r'href="([^"]+exhibit99[^"]*\.html)"',
            r'href="([^"]+ex-99[^"]*\.htm)"',
        ]

        for pat in patterns:
            m = re.search(pat, filing_primary_html, flags=re.IGNORECASE)
            if m:
                href = m.group(1)
                return urljoin(filing_base_dir_url, href)

        raise ValueError("Could not find an Exhibit 99.* link in the 10-D HTML")
