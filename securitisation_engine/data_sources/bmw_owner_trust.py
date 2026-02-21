from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from .sec_edgar import SECEdgarClient


_MONEY = r"([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2})"
_DATE = r"([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})"


def _to_float(x: str) -> float:
    x = x.strip()
    if x in ("-", "—", "–", ""):
        return 0.0
    return float(x.replace(",", ""))


def _parse_date(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unparseable date: {s}")


def _strip_html(html: str) -> str:
    # crude but effective for these statements
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = txt.replace("\xa0", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _find_date(txt: str, label: str) -> datetime:
    m = re.search(re.escape(label) + r"\s*" + _DATE, txt, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not find date for label: {label}")
    return _parse_date(m.group(1))


def _find_amount(txt: str, label: str) -> float:
    m = re.search(re.escape(label) + r".{0,200}?" + _MONEY, txt, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not find amount for label: {label}")
    return _to_float(m.group(1))


def _find_amounts_after(txt: str, label: str, n: int) -> List[float]:
    # find label, then capture next n monetary values
    idx = txt.lower().find(label.lower())
    if idx < 0:
        raise ValueError(f"Could not find label: {label}")
    window = txt[idx : idx + 2000]
    vals = re.findall(_MONEY, window)
    if len(vals) < n:
        raise ValueError(f"Expected {n} amounts after '{label}', found {len(vals)}")
    return [_to_float(v) for v in vals[:n]]


@dataclass(frozen=True)
class BMWParsedStatement:
    payment_date: datetime
    accrued_interest_date: datetime
    collection_period_ending: datetime

    day_count_fraction_act_360: float

    pool_balance_end: float
    reserve_opening: float

    total_available_interest: float
    total_available_principal: float

    fees: Dict[str, float]

    # tranche info keyed by name like "A-1", "A-2a"...
    tranche_coupon: Dict[str, float]          # annual % as decimal, from statement interest rate
    tranche_begin_balance: Dict[str, float]
    tranche_principal_paid: Dict[str, float]
    tranche_end_balance: Dict[str, float]


def parse_bmw_exhibit_99_1(html: str) -> BMWParsedStatement:
    txt = _strip_html(html)

    payment_date = _find_date(txt, "Current Payment Date:")
    accrued_interest_date = _find_date(txt, "Accrued Interest Date:")
    collection_period_ending = _find_date(txt, "Collection Period Ending:")

    dcf = (payment_date.date() - accrued_interest_date.date()).days / 360.0

    # Pool Balance table has three columns: Beginning / End / Initial
    pool_bal_begin, pool_bal_end, _pool_bal_initial = _find_amounts_after(txt, "Pool Balance", 3)

    # Reserve Account also shows three columns; opening is the first value
    reserve_open, _reserve_end, _reserve_initial = _find_amounts_after(txt, "Reserve Account", 3)

    total_avail_int = _find_amount(txt, "Total Available Interest")
    total_avail_prn = _find_amount(txt, "Total Available Principal")

    fees = {}
    # Common BMW line items in Exhibit 99.1
    for fee_label in [
        "Servicing Fees",
        "Non-recoverable Servicer Advance Reimbursement",
        "Amounts paid to Indenture Trustee, Owner Trustee and Asset Representations Reviewer (subject to annual cap)",
        "Amounts paid to Indenture Trustee, Owner Trustee and Asset Representations Reviewer (not subject to annual cap)",
    ]:
        try:
            fees[fee_label] = _find_amount(txt, fee_label)
        except ValueError:
            # not always present / sometimes 0 with weird formatting
            continue

    # Interest table parsing (Class ... Notes, rate %, payment $)
    # Example text chunk:
    # "Class A-1 Notes 4.40100 % $ 853,722.95"
    interest_pat = re.compile(
        r"(Class\s+([A-Za-z0-9\-]+[a-z]?)\s+Notes)\s+([0-9]+\.[0-9]+)\s*%\s*\$\s*" + _MONEY,
        flags=re.IGNORECASE,
    )
    tranche_coupon: Dict[str, float] = {}
    for _full, short, rate, pay in interest_pat.findall(txt):
        name = short.strip()
        tranche_coupon[name] = float(rate) / 100.0

    # Principal table parsing (Class ... Notes $ begin $ pay $ end)
    # Example:
    # "Class A-1 Notes $ 225,271,573.82 $ 63,394,229.81 $ 161,877,344.01"
    principal_pat = re.compile(
        r"(Class\s+([A-Za-z0-9\-]+[a-z]?)\s+Notes)\s*\$\s*"
        + _MONEY
        + r"\s*\$\s*("
        + _MONEY
        + r"|[-—–])\s*\$\s*"
        + _MONEY,
        flags=re.IGNORECASE,
    )

    begin_bal: Dict[str, float] = {}
    prn_paid: Dict[str, float] = {}
    end_bal: Dict[str, float] = {}

    for _full, short, b, p, e in principal_pat.findall(txt):
        name = short.strip()
        begin_bal[name] = _to_float(b)
        prn_paid[name] = _to_float(p)
        end_bal[name] = _to_float(e)

    if not begin_bal:
        raise ValueError("Could not parse tranche principal table (begin/pay/end) from Exhibit 99.1")

    return BMWParsedStatement(
        payment_date=payment_date,
        accrued_interest_date=accrued_interest_date,
        collection_period_ending=collection_period_ending,
        day_count_fraction_act_360=dcf,
        pool_balance_end=pool_bal_end,
        reserve_opening=reserve_open,
        total_available_interest=total_avail_int,
        total_available_principal=total_avail_prn,
        fees=fees,
        tranche_coupon=tranche_coupon,
        tranche_begin_balance=begin_bal,
        tranche_principal_paid=prn_paid,
        tranche_end_balance=end_bal,
    )


def build_engine_input_excel(parsed: BMWParsedStatement, out_xlsx: str) -> None:
    """
    Writes your engine's expected input workbook:
      - Deal (key/value)
      - Fees (fee_name/amount)
      - Tranches (name/rank/coupon/opening_balance/is_residual)

    Notes:
      - BMW has many note classes; we load each class as a tranche.
      - Add a residual "Certificates" tranche for convenience.
      - Tranche rank order is inferred: A-1, A-2a, A-2b, A-3, A-4, ... then Certificates.
    """
    # Build tranches list
    def rank_key(n: str) -> Tuple[int, str]:
        # crude ordering for common BMW naming
        # A-1 < A-2a < A-2b < A-3 < A-4 ...; fallback lexicographic
        m = re.match(r"^A-(\d+)([a-z]?)$", n, flags=re.IGNORECASE)
        if m:
            num = int(m.group(1))
            suf = m.group(2) or ""
            # 'a' before 'b' etc
            return (num, suf)
        return (999, n.lower())

    note_names = sorted(parsed.tranche_begin_balance.keys(), key=rank_key)

    tr_rows = []
    for i, name in enumerate(note_names, start=1):
        coupon = parsed.tranche_coupon.get(name, 0.0)  # sometimes fixed notes; should parse fine
        tr_rows.append(
            {
                "name": name,
                "rank": i,
                "coupon": coupon,
                "opening_balance": float(parsed.tranche_begin_balance[name]),
                "is_residual": False,
            }
        )

    # Residual
    tr_rows.append(
        {"name": "Certificates", "rank": 99, "coupon": 0.0, "opening_balance": 0.0, "is_residual": True}
    )

    tranches_df = pd.DataFrame(tr_rows)

    deal_df = pd.DataFrame(
        [
            {"key": "payment_date", "value": parsed.payment_date.strftime("%Y-%m-%d")},
            {"key": "day_count_fraction", "value": parsed.day_count_fraction_act_360},
            # BMW statement doesn't publish OC/IC triggers in a simple way; keep these inert
            {"key": "oc_trigger", "value": 0.0},
            {"key": "ic_trigger", "value": 0.0},
            # reserve target: BMW often holds a specified balance; use opening as a proxy
            {"key": "reserve_target", "value": parsed.reserve_opening},
            {"key": "reserve_opening", "value": parsed.reserve_opening},
            {"key": "collateral_balance", "value": parsed.pool_balance_end},
            {"key": "interest_collections", "value": parsed.total_available_interest},
            {"key": "principal_collections", "value": parsed.total_available_principal},
        ]
    )

    fees_df = pd.DataFrame(
        [{"fee_name": k, "amount": float(v)} for k, v in parsed.fees.items()]
    )
    if fees_df.empty:
        fees_df = pd.DataFrame([{"fee_name": "Servicing Fees", "amount": 0.0}])

    with pd.ExcelWriter(out_xlsx) as xw:
        deal_df.to_excel(xw, sheet_name="Deal", index=False)
        fees_df.to_excel(xw, sheet_name="Fees", index=False)
        tranches_df.to_excel(xw, sheet_name="Tranches", index=False)


def fetch_latest_bmw_exhibit991_to_input_xlsx(
    cik: str | int,
    user_agent: str,
    out_xlsx: str,
) -> Tuple[str, str]:
    """
    Returns (10-D primary doc URL, Exhibit 99.1 URL) and writes out_xlsx.
    """
    client = SECEdgarClient(user_agent=user_agent)

    filing = client.latest_filing(cik, form="10-D")
    ten_d_html = client.get_text(filing.primary_doc_url)
    exhibit_url = client.find_first_exhibit_url(ten_d_html, filing.base_dir_url)

    exhibit_html = client.get_text(exhibit_url)
    parsed = parse_bmw_exhibit_99_1(exhibit_html)
    build_engine_input_excel(parsed, out_xlsx=out_xlsx)

    return filing.primary_doc_url, exhibit_url
