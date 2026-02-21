from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .models import PeriodInputs, WaterfallResults
from .config import EngineConfig


def build_available_funds(period: PeriodInputs, res: WaterfallResults) -> pd.DataFrame:
    rows = [
        {"section": "Interest", "line": "Interest Collections", "amount": period.interest_collections},
        {"section": "Interest", "line": "Reserve Draw", "amount": res.reserve_draw},
        {"section": "Interest", "line": "Total Interest Available", "amount": period.interest_collections + res.reserve_draw},
        {"section": "Principal", "line": "Principal Collections", "amount": period.principal_collections},
        {"section": "Principal", "line": "Turbo Principal from Interest (IC breach)", "amount": sum(
            u["amount"] for u in res.interest_uses if str(u["step"]).startswith("IC Turbo Principal")
        )},
        {"section": "Principal", "line": "Total Principal Available", "amount": period.principal_collections + sum(
            u["amount"] for u in res.interest_uses if str(u["step"]).startswith("IC Turbo Principal")
        )},
        {"section": "Reserve", "line": "Reserve Opening", "amount": period.reserve_opening},
        {"section": "Reserve", "line": "Reserve Replenish", "amount": res.reserve_replenish},
        {"section": "Reserve", "line": "Reserve Closing", "amount": res.reserve_closing},
    ]
    return pd.DataFrame(rows)


def build_priority_of_payments(res: WaterfallResults) -> Tuple[pd.DataFrame, pd.DataFrame]:
    int_df = pd.DataFrame(res.interest_uses)
    prn_df = pd.DataFrame(res.principal_uses)
    # ensure consistent column order
    for df in (int_df, prn_df):
        if not df.empty:
            df = df[["step", "amount"]]
    return int_df[["step", "amount"]], prn_df[["step", "amount"]]


def build_note_rollforward(cfg: EngineConfig, res: WaterfallResults, tranche_opening: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for t in cfg.sorted_tranches():
        open_bal = float(tranche_opening.get(t.name, 0.0))
        prn_paid = float(res.tranche_principal_paid.get(t.name, 0.0))
        close_bal = float(res.tranche_balance_closing.get(t.name, open_bal))
        due_int = float(res.tranche_interest_due.get(t.name, 0.0))
        paid_int = float(res.tranche_interest_paid.get(t.name, 0.0))
        sf_close = float(res.tranche_interest_shortfall_closing.get(t.name, 0.0))
        rows.append({
            "Tranche": t.name,
            "Opening Balance": open_bal,
            "Principal Paid": prn_paid,
            "Closing Balance": close_bal,
            "Interest Due": due_int,
            "Interest Paid": paid_int,
            "Interest Shortfall (Closing)": sf_close,
        })
    return pd.DataFrame(rows)


def build_investor_summary(period: PeriodInputs, res: WaterfallResults) -> pd.DataFrame:
    rows = [
        {"metric": "Payment Date", "value": period.payment_date},
        {"metric": "OC Ratio", "value": res.oc_ratio},
        {"metric": "OC Trigger Breached", "value": str(res.oc_breached)},
        {"metric": "IC Ratio", "value": res.ic_ratio},
        {"metric": "IC Trigger Breached", "value": str(res.ic_breached)},
        {"metric": "Reserve Closing", "value": res.reserve_closing},
    ]
    return pd.DataFrame(rows)
