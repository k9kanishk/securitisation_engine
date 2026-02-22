from __future__ import annotations

import itertools
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from dateutil import parser as dtparser

from .sec_edgar import SECEdgarClient


_MONEY = r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)"  # allow optional decimals
_DATE = r"([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})"


def _to_float(x: str) -> float:
    x = str(x).strip()
    if x in ("-", "—", "–", "", "nan", "None"):
        return 0.0
    # parentheses for negatives
    neg = False
    if x.startswith("(") and x.endswith(")"):
        neg = True
        x = x[1:-1].strip()
    x = x.replace("$", "").replace(",", "").strip()
    if x == "":
        return 0.0
    v = float(x)
    return -v if neg else v


def _try_parse_money(v):
    """
    Returns float if value looks like a number/money, else None.
    """
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "-", "—", "–", "nan", "None"):
        return None
    # fast check: must contain a digit
    if not any(ch.isdigit() for ch in s):
        return None
    try:
        return _to_float(s)
    except Exception:
        return None


def _parse_date(s: str) -> datetime:
    s = s.strip()
    # Try known formats first (fast + deterministic)
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # Fallback: dateutil handles lots of weirdness
    try:
        return dtparser.parse(s, dayfirst=False)
    except Exception as e:
        raise ValueError(f"Unparseable date: {s}") from e


def _strip_html(html: str) -> str:
    # crude but effective for these statements
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = txt.replace("\xa0", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt




def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Handles MultiIndex columns from read_html
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in col if str(x) != "nan"]).strip()
            for col in df.columns.values
        ]
    df.columns = [str(c).strip() if str(c).strip() != "" else f"col_{i}" for i, c in enumerate(df.columns)]
    return df


def _table_cells(df: pd.DataFrame):
    # Flatten multi-index columns if any, and drop empty rows
    df = _flatten_cols(df).dropna(how="all")
    return df


def _row_contains(row, label: str) -> bool:
    lab = label.lower()
    return any(lab in str(x).lower() for x in row.values)


def _row_numbers(row, min_abs: float = 0.0) -> list[float]:
    nums = []
    for x in row.values:
        v = _try_parse_money(x)
        if v is None:
            continue
        if abs(v) >= min_abs:
            nums.append(float(v))
    return nums


def _find_row_amount_in_tables(tables: list[pd.DataFrame], label: str, *, min_abs: float = 1000.0) -> float:
    """
    Finds a row containing 'label' in ANY table and returns a numeric amount from that row.
    We default min_abs=1000 to avoid picking up ratios / per-$1000 / period numbers.
    """
    for df in tables:
        df = _table_cells(df)
        for _, row in df.iterrows():
            if _row_contains(row, label):
                nums = _row_numbers(row, min_abs=min_abs)
                if nums:
                    # usually the amount is the last/only big number
                    return nums[-1]
    raise ValueError(f"Could not find numeric amount for '{label}' in HTML tables")


def _find_row_triplet_in_tables(tables: list[pd.DataFrame], label: str, *, min_abs: float = 1000.0) -> list[float]:
    """
    For rows like Pool Balance / Reserve Account that show Initial / Beginning / End.
    Returns first 3 big numbers found.
    """
    for df in tables:
        df = _table_cells(df)
        for _, row in df.iterrows():
            if _row_contains(row, label):
                nums = _row_numbers(row, min_abs=min_abs)
                if len(nums) >= 3:
                    return nums[:3]
    raise ValueError(f"Could not find 3 numeric values for '{label}' in HTML tables")


def _find_interest_rate_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """
    Finds the 'Interest Distributable Amount' table.
    """
    for df in tables:
        df2 = _table_cells(df)
        # detect table by presence of this header text in any cell
        if df2.astype(str).apply(lambda col: col.str.contains("Interest Distributable Amount", case=False, na=False)).any().any():
            return df2
    return None


def _parse_tranche_coupons_from_interest_table(df: pd.DataFrame) -> dict[str, float]:
    """
    Parses coupon rates from the Interest Distributable Amount table.
    Heuristic: in each tranche row, the first small number (0<rate<30) is the rate %.
    """
    name_pat = re.compile(r"Class\s+([A-Za-z0-9\-]+[a-z]?)\s+Notes", flags=re.IGNORECASE)
    coupons = {}

    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row.values)
        m = name_pat.search(row_text)
        if not m:
            continue
        name = m.group(1).strip()

        # collect small numerics (rates/payments/per1000)
        small = []
        for x in row.values:
            v = _try_parse_money(x)
            if v is None:
                continue
            v = float(v)
            if 0.0 < v < 30.0:
                small.append(v)

        if small:
            coupons[name] = small[0] / 100.0  # percent -> decimal
    return coupons


def _parse_tranche_interest_paid_from_interest_table(df: pd.DataFrame) -> dict[str, float]:
    """
    Parses the 'Current Payment' amounts from the Interest Distributable Amount table.
    Heuristic: for each tranche row, the largest number >= 1,000 is the payment amount.
    """
    name_pat = re.compile(r"Class\s+([A-Za-z0-9\-]+[a-z]?)\s+Notes", flags=re.IGNORECASE)
    pays: Dict[str, float] = {}

    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row.values)
        m = name_pat.search(row_text)
        if not m:
            continue
        name = m.group(1).strip()

        nums = []
        for x in row.values:
            v = _try_parse_money(x)
            if v is None:
                continue
            v = float(v)
            if abs(v) >= 1000.0:
                nums.append(v)

        pays[name] = float(max(nums, key=abs)) if nums else 0.0

    return pays


def _best_col_by_keywords(cols_lower, include_any, include_all=None, exclude_any=None):
    include_all = include_all or []
    exclude_any = exclude_any or []
    best_i, best_score = None, -1

    for i, c in enumerate(cols_lower):
        if any(x in c for x in exclude_any):
            continue
        if include_all and not all(x in c for x in include_all):
            continue
        score = sum(1 for x in include_any if x in c)
        if score > best_score:
            best_i, best_score = i, score

    return best_i


def _summarize_tables(html: str) -> str:
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception as e:
        return f"read_html failed: {e}"

    parts = []
    for i, df in enumerate(tables[:12]):
        df = _flatten_cols(df)
        cols = list(df.columns)
        parts.append(f"[{i}] shape={df.shape} cols={cols[:10]}")
    return " | ".join(parts)


def _parse_tranche_principal_from_tables(html: str):
    """
    Robust principal roll-forward parser using HTML tables.

    Works even when tables have no usable headers (columns named '0','1',...).
    It infers (Begin, PrincipalPaid, End) by minimizing: |(Begin - Paid) - End|.
    """
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    name_pat = re.compile(r"Class\s+([A-Za-z0-9\-]+[a-z]?)\s+Notes", flags=re.IGNORECASE)

    best = None  # (score, df, name_col, begin_idx, prn_idx, end_idx)

    for raw in tables:
        if raw is None or raw.empty:
            continue

        df = raw.copy()
        df = _flatten_cols(df).dropna(how="all")
        if df.empty or df.shape[1] < 3:
            continue

        # Find tranche rows and which column contains the tranche label most often
        str_df = df.astype(str)
        match_counts = {col: str_df[col].str.contains(name_pat, na=False).sum() for col in df.columns}
        name_col = max(match_counts, key=match_counts.get)
        if match_counts[name_col] < 2:
            continue

        # Collect tranche rows
        tranche_rows = []
        tranche_names = []
        for idx, row in df.iterrows():
            m = name_pat.search(str(row[name_col]))
            if not m:
                # scan row cells
                for cell in row.values:
                    mm = name_pat.search(str(cell))
                    if mm:
                        m = mm
                        break
            if m:
                tranche_rows.append(idx)
                tranche_names.append(m.group(1).strip())

        if len(tranche_rows) < 2:
            continue

        # Identify numeric columns (by content), not by header
        col_vals = {}
        for j, col in enumerate(df.columns):
            vals = []
            for idx in tranche_rows:
                v = _try_parse_money(df.loc[idx, col])
                if v is not None and math.isfinite(v):
                    vals.append(v)
                else:
                    vals.append(None)
            # keep columns that have enough numeric values
            if sum(x is not None for x in vals) >= 2:
                col_vals[j] = vals

        if len(col_vals) < 3:
            continue

        col_indices = list(col_vals.keys())

        # Score permutations of triples: choose (begin, principal, end) that best fit begin - principal = end
        for a, b, c in itertools.combinations(col_indices, 3):
            for begin_idx, prn_idx, end_idx in itertools.permutations([a, b, c], 3):
                err_sum = 0.0
                used = 0
                bad = 0

                for r_i in range(len(tranche_rows)):
                    vb = col_vals[begin_idx][r_i]
                    vp = col_vals[prn_idx][r_i]
                    ve = col_vals[end_idx][r_i]
                    if vb is None or vp is None or ve is None:
                        continue

                    used += 1
                    # sanity: balances should be non-negative mostly; principal paid non-negative mostly
                    if vb < -1e-6 or ve < -1e-6:
                        bad += 1
                    # equation error
                    err = abs((vb - vp) - ve)
                    err_sum += err

                if used < 2:
                    continue

                avg_err = err_sum / used
                # penalize weird assignments
                score = avg_err + (bad * 1e6) + (1e3 / used)

                if best is None or score < best[0]:
                    best = (score, df, name_col, begin_idx, prn_idx, end_idx, tranche_rows, tranche_names)

    if best is None:
        return {}, {}, {}

    _, df, name_col, begin_idx, prn_idx, end_idx, tranche_rows, tranche_names = best

    begin_bal, prn_paid, end_bal = {}, {}, {}
    for idx, name in zip(tranche_rows, tranche_names):
        begin_bal[name] = _to_float(df.iloc[df.index.get_loc(idx), begin_idx])
        prn_paid[name] = _to_float(df.iloc[df.index.get_loc(idx), prn_idx])
        end_bal[name] = _to_float(df.iloc[df.index.get_loc(idx), end_idx])

    # Sometimes BMW prints principal as negative (rare). Force principal paid positive if it fixes identity.
    # If principal is negative but begin - (-p) moves away from end, flip sign.
    for k in list(begin_bal.keys()):
        b = begin_bal[k]
        p = prn_paid[k]
        e = end_bal[k]
        if abs((b - p) - e) > abs((b - abs(p)) - e):
            prn_paid[k] = abs(p)

    return begin_bal, prn_paid, end_bal

def _find_date(txt: str, label: str) -> datetime:
    # Accept label with/without colon (BMW sometimes changes)
    label_variants = [label, label.rstrip(":"), label.rstrip(":") + " :"]

    # Date can be numeric (MM/DD/YYYY) OR month-name (January 31, 2026)
    date_pat = r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})"

    for lab in label_variants:
        m = re.search(re.escape(lab) + r"\s*" + date_pat, txt, flags=re.IGNORECASE)
        if m:
            return _parse_date(m.group(1))

    raise ValueError(f"Could not find date for label: {label}")


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
    tranche_interest_paid_exhibit: Dict[str, float]  # NEW: current payment from interest table
    tranche_begin_balance: Dict[str, float]
    tranche_principal_paid: Dict[str, float]
    tranche_end_balance: Dict[str, float]


def parse_bmw_exhibit_99_1(html: str) -> BMWParsedStatement:
    txt = _strip_html(html)
    tables = pd.read_html(io.StringIO(html), flavor="lxml")

    payment_date = _find_date(txt, "Current Payment Date:")
    accrued_interest_date = _find_date(txt, "Accrued Interest Date:")
    try:
        collection_period_ending = _find_date(txt, "Collection Period Ending:")
    except ValueError:
        # BMW format changes; not needed for engine calcs, so fallback safely
        collection_period_ending = payment_date

    dcf = (payment_date.date() - accrued_interest_date.date()).days / 360.0

    # --- Balances table (Initial / Beginning / End) ---
    pool_triplet = _find_row_triplet_in_tables(tables, "Pool Balance", min_abs=1000.0)
    pool_bal_initial, pool_bal_begin, pool_bal_end = pool_triplet[0], pool_triplet[1], pool_triplet[2]

    reserve_triplet = _find_row_triplet_in_tables(tables, "Reserve Account", min_abs=1000.0)
    reserve_initial, reserve_begin, reserve_end = reserve_triplet[0], reserve_triplet[1], reserve_triplet[2]

    # --- Available funds totals ---
    total_avail_int = _find_row_amount_in_tables(tables, "Total Available Interest", min_abs=1000.0)
    total_avail_prn = _find_row_amount_in_tables(tables, "Total Available Principal", min_abs=1000.0)

    # --- Fees from Distributions list ---
    fees = {}
    for fee_label in [
        "Servicing Fees",
        "Non-recoverable Servicer Advance Reimbursement",
        "Amounts paid to Indenture Trustee, Owner Trustee and Asset Representations Reviewer (subject to annual cap)",
        "Amounts paid to Indenture Trustee, Owner Trustee and Asset Representations Reviewer (not subject to annual cap)",
    ]:
        try:
            fees[fee_label] = _find_row_amount_in_tables(tables, fee_label, min_abs=0.0)  # can be 0.00
        except ValueError:
            continue

    # --- Coupons from Interest Distributable table ---
    interest_tbl = _find_interest_rate_table(tables)
    tranche_coupon = _parse_tranche_coupons_from_interest_table(interest_tbl) if interest_tbl is not None else {}
    tranche_interest_paid_exhibit = _parse_tranche_interest_paid_from_interest_table(interest_tbl) if interest_tbl is not None else {}

    reserve_opening = reserve_begin
    pool_balance_end = pool_bal_end

    # --- Tranche principal roll-forward: use HTML tables first (robust) ---
    begin_bal, prn_paid, end_bal = _parse_tranche_principal_from_tables(html)

    # Fallback to old regex approach if table parsing fails (kept as backup)
    if not begin_bal:
        principal_pat = re.compile(
            r"(Class\s+([A-Za-z0-9\-]+[a-z]?)\s+Notes)\s*\$?\s*"
            + _MONEY
            + r"\s*\$?\s*("
            + _MONEY
            + r"|[-—–])\s*\$?\s*"
            + _MONEY,
            flags=re.IGNORECASE,
        )

        begin_bal = {}
        prn_paid = {}
        end_bal = {}

        for _full, short, b, p, e in principal_pat.findall(txt):
            name = short.strip()
            begin_bal[name] = _to_float(b)
            prn_paid[name] = _to_float(p)
            end_bal[name] = _to_float(e)

    if not begin_bal:
        table_summary = _summarize_tables(html)
        raise ValueError(
            "Could not parse tranche principal table (begin/pay/end) from Exhibit 99.1. "
            f"Table scan summary: {table_summary}"
        )

    return BMWParsedStatement(
        payment_date=payment_date,
        accrued_interest_date=accrued_interest_date,
        collection_period_ending=collection_period_ending,
        day_count_fraction_act_360=dcf,
        pool_balance_end=pool_balance_end,
        reserve_opening=reserve_opening,
        total_available_interest=total_avail_int,
        total_available_principal=total_avail_prn,
        fees=fees,
        tranche_coupon=tranche_coupon,
        tranche_interest_paid_exhibit=tranche_interest_paid_exhibit,
        tranche_begin_balance=begin_bal,
        tranche_principal_paid=prn_paid,
        tranche_end_balance=end_bal,
    )


def build_engine_input_excel(parsed: BMWParsedStatement, out_xlsx: str) -> None:
    """
    Writes engine input workbook:
      - Deal (key/value)
      - Fees (fee_name/amount)
      - Tranches (name/rank/coupon/opening_balance/is_residual/dcf)
    """
    def rank_key(n: str) -> Tuple[int, str]:
        m = re.match(r"^A-(\d+)([a-z]?)$", n, flags=re.IGNORECASE)
        if m:
            num = int(m.group(1))
            suf = m.group(2) or ""
            return (num, suf)
        return (999, n.lower())

    def rank_from_name(n: str) -> int:
        m = re.match(r"^A-(\d+)", n, flags=re.IGNORECASE)
        return int(m.group(1)) if m else 999

    note_names = sorted(parsed.tranche_begin_balance.keys(), key=rank_key)

    # infer tranche-specific dcf from Exhibit interest payments
    act_dcf = float(parsed.day_count_fraction_act_360)
    dcf_30_360 = 30.0 / 360.0

    def snap_dcf(x: float) -> float:
        if abs(x - dcf_30_360) < 1e-4:
            return dcf_30_360
        if abs(x - act_dcf) < 1e-4:
            return act_dcf
        return x

    dcf_by_tranche: Dict[str, float] = {}
    for name in note_names:
        pay = float(parsed.tranche_interest_paid_exhibit.get(name, 0.0))
        bal = float(parsed.tranche_begin_balance.get(name, 0.0))
        cpn = float(parsed.tranche_coupon.get(name, 0.0))
        if pay > 0 and bal > 0 and cpn > 0:
            implied = pay / (bal * cpn)
            dcf_by_tranche[name] = snap_dcf(implied)
        else:
            dcf_by_tranche[name] = act_dcf

    # principal cap to notes = sum of Exhibit note principal payments
    principal_to_notes_cap = float(sum(float(parsed.tranche_principal_paid.get(n, 0.0)) for n in note_names))

    tr_rows = []
    for name in note_names:
        tr_rows.append(
            {
                "name": name,
                "rank": rank_from_name(name),  # A-2a and A-2b both -> 2
                "coupon": float(parsed.tranche_coupon.get(name, 0.0)),
                "opening_balance": float(parsed.tranche_begin_balance[name]),
                "is_residual": False,
                "dcf": float(dcf_by_tranche.get(name, act_dcf)),
            }
        )

    # Residual
    tr_rows.append(
        {"name": "Certificates", "rank": 99, "coupon": 0.0, "opening_balance": 0.0, "is_residual": True, "dcf": None}
    )
    tranches_df = pd.DataFrame(tr_rows)

    deal_df = pd.DataFrame(
        [
            {"key": "payment_date", "value": parsed.payment_date.strftime("%Y-%m-%d")},
            {"key": "day_count_fraction", "value": act_dcf},  # base fallback
            {"key": "oc_trigger", "value": 0.0},
            {"key": "ic_trigger", "value": 0.0},
            {"key": "reserve_target", "value": parsed.reserve_opening},
            {"key": "reserve_opening", "value": parsed.reserve_opening},
            {"key": "collateral_balance", "value": parsed.pool_balance_end},
            {"key": "interest_collections", "value": parsed.total_available_interest},
            {"key": "principal_collections", "value": parsed.total_available_principal},
            {"key": "principal_to_notes_cap", "value": principal_to_notes_cap},  # NEW
        ]
    )

    fees_df = pd.DataFrame([{"fee_name": k, "amount": float(v)} for k, v in parsed.fees.items()])
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
