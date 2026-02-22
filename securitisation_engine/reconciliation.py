from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def _r2(x: float) -> float:
    return float(round(float(x or 0.0) + 1e-9, 2))


def _get_af(df: pd.DataFrame, line: str) -> float:
    s = df.loc[df["line"] == line, "amount"]
    if s.empty:
        raise KeyError(f"Available Funds missing line: {line}")
    return _r2(float(s.iloc[0]))


def _get_pop(df: pd.DataFrame, step: str) -> float:
    s = df.loc[df["step"] == step, "amount"]
    if s.empty:
        return 0.0
    return _r2(float(s.iloc[0]))


def _sum_pop_interest_distributions(df: pd.DataFrame) -> float:
    # Reserve Draw is a funding source, not a distribution line
    return _r2(
        float(df.loc[~df["step"].astype(str).str.contains("Reserve Draw", case=False, na=False), "amount"].sum())
    )


def build_reconciliation(
    engine_dfs: Dict[str, pd.DataFrame],
    exhibit: Dict[str, Any],
) -> pd.DataFrame:
    """
    engine_dfs: outputs from runner (Available Funds, PoP sheets, Note Rollforward, etc.)
    exhibit: dict from BMW Exhibit 99.1 parsing (see bmw_owner_trust.parse_bmw_exhibit_recon_metrics)
    """
    af = engine_dfs["Available Funds"]
    pop_i = engine_dfs["Priority of Payments - Interest"]
    pop_p = engine_dfs["Priority of Payments - Principal"]
    roll = engine_dfs["Note Rollforward"]

    # ----- Engine totals -----
    eng_av_int = _get_af(af, "Total Interest Available")
    eng_av_prn = _get_af(af, "Total Principal Available")
    eng_total_funds = _r2(eng_av_int + eng_av_prn)

    eng_fee_serv = _get_pop(pop_i, "Fee: Servicing Fees")
    eng_fee_nonrec = _get_pop(pop_i, "Fee: Non-recoverable Servicer Advance Reimbursement")

    # tranche rollforward totals
    roll = roll.copy()
    roll["Tranche"] = roll["Tranche"].astype(str)

    residual_name = "Certificates"
    note_roll = roll.loc[roll["Tranche"] != residual_name].copy()
    res_roll = roll.loc[roll["Tranche"] == residual_name].copy()

    eng_note_int = _r2(float(note_roll["Interest Paid"].sum()))
    eng_note_prn = _r2(float(note_roll["Principal Paid"].sum()))

    eng_res_int = _r2(float(res_roll["Interest Paid"].sum())) if not res_roll.empty else 0.0
    eng_res_prn = _r2(float(res_roll["Principal Paid"].sum())) if not res_roll.empty else 0.0
    eng_cert_dist = _r2(eng_res_int + eng_res_prn)

    # total distributions: interest distributions (fees + tranche interest + residual interest + reserve deposits etc)
    eng_dist_interest = _sum_pop_interest_distributions(pop_i)
    eng_dist_principal = _r2(float(pop_p["amount"].sum()))
    eng_total_distributions = _r2(eng_dist_interest + eng_dist_principal)

    # reserve checks
    eng_res_open = _get_af(af, "Reserve Opening")
    eng_res_close = _get_af(af, "Reserve Closing")

    # ----- Exhibit totals -----
    ex_av_int = _r2(float(exhibit.get("total_available_interest", 0.0)))
    ex_av_prn = _r2(float(exhibit.get("total_available_principal", 0.0)))
    ex_total_funds = _r2(float(exhibit.get("total_available_funds", ex_av_int + ex_av_prn)))

    ex_serv_fee = _r2(float(exhibit.get("servicing_fees", 0.0)))
    ex_nonrec = _r2(float(exhibit.get("nonrecoverable_servicer_adv_reimb", 0.0)))

    ex_note_int = _r2(float(exhibit.get("note_interest_total", 0.0)))
    ex_note_prn = _r2(float(exhibit.get("note_principal_total", 0.0)))
    ex_cert_dist = _r2(float(exhibit.get("certificate_distribution", 0.0)))
    ex_total_dist = _r2(float(exhibit.get("total_distributions", 0.0)))

    ex_res_open = _r2(float(exhibit.get("reserve_opening", 0.0)))
    ex_res_close = _r2(float(exhibit.get("reserve_closing", 0.0)))

    # ----- Rows -----
    rows = []

    def add(section: str, metric: str, ex: Optional[float], eng: Optional[float]):
        exv = "" if ex is None else _r2(ex)
        engv = "" if eng is None else _r2(eng)
        diff = "" if (ex is None or eng is None) else _r2(eng - ex)
        ok = "" if (ex is None or eng is None) else ("PASS" if abs(diff) <= 0.01 else "FAIL")
        rows.append(
            {"Section": section, "Metric": metric, "Exhibit": exv, "Engine": engv, "Diff": diff, "Status": ok}
        )

    # Control totals
    add("Control Totals", "Total Available Interest", ex_av_int, eng_av_int)
    add("Control Totals", "Total Available Principal", ex_av_prn, eng_av_prn)
    add("Control Totals", "Total Available Funds", ex_total_funds, eng_total_funds)
    add("Distributions", "Servicing Fees", ex_serv_fee, eng_fee_serv)
    add("Distributions", "Non-recoverable Servicer Advance Reimbursement", ex_nonrec, eng_fee_nonrec)
    add("Distributions", "Total Note Interest Paid", ex_note_int, eng_note_int)
    add("Distributions", "Total Note Principal Paid", ex_note_prn, eng_note_prn)
    add("Distributions", "Certificates Distribution (Interest+Principal)", ex_cert_dist, eng_cert_dist)
    add("Distributions", "Total Distributions", ex_total_dist, eng_total_distributions)
    add("Reserve", "Reserve Opening", ex_res_open, eng_res_open)
    add("Reserve", "Reserve Closing", ex_res_close, eng_res_close)

    # Tranche-level checks
    ex_int_by = exhibit.get("tranche_interest_paid", {}) or {}
    ex_prn_by = exhibit.get("tranche_principal_paid", {}) or {}

    # Build engine maps from rollforward
    eng_int_by = {str(r["Tranche"]): _r2(float(r["Interest Paid"])) for _, r in roll.iterrows()}
    eng_prn_by = {str(r["Tranche"]): _r2(float(r["Principal Paid"])) for _, r in roll.iterrows()}

    # Only reconcile BMW notes (A- classes); ignore Certificate line here (already checked above)
    for name, exv in sorted(ex_int_by.items()):
        if str(name).strip().lower().startswith("a-"):
            add("Tranche Interest", f"Interest Paid - {name}", _r2(float(exv)), eng_int_by.get(name, 0.0))

    for name, exv in sorted(ex_prn_by.items()):
        if str(name).strip().lower().startswith("a-"):
            add("Tranche Principal", f"Principal Paid - {name}", _r2(float(exv)), eng_prn_by.get(name, 0.0))

    return pd.DataFrame(rows)
