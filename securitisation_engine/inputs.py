from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from openpyxl import load_workbook

from .config import EngineConfig
from .models import DealConfig, PeriodInputs, TrancheConfig, TrancheState


def _read_table(path: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def read_engine_inputs(excel_path: str) -> Tuple[EngineConfig, PeriodInputs, Dict[str, TrancheState]]:
    """
    Expected sheets:
      - Deal (key/value table)
      - Tranches (columns: name, rank, coupon, opening_balance, is_residual)
    """
    deal_df = _read_table(excel_path, "Deal")
    if set(deal_df.columns) != {"key", "value"}:
        raise ValueError("Deal sheet must have columns: key, value")

    deal_kv = {str(k): float(v) if str(k) not in ("payment_date",) else v for k, v in zip(deal_df["key"], deal_df["value"])}

    fees_df = _read_table(excel_path, "Fees")
    if not {"fee_name", "amount"}.issubset(fees_df.columns):
        raise ValueError("Fees sheet must have columns: fee_name, amount")
    fees = {str(r["fee_name"]): float(r["amount"]) for _, r in fees_df.iterrows()}

    tr_df = _read_table(excel_path, "Tranches")
    required = {"name", "rank", "coupon", "opening_balance", "is_residual"}
    if not required.issubset(tr_df.columns):
        raise ValueError(f"Tranches sheet must include columns: {sorted(required)}")

    tranches: List[TrancheConfig] = []
    tranche_states: Dict[str, TrancheState] = {}

    for _, r in tr_df.iterrows():
        name = str(r["name"])
        tranches.append(
            TrancheConfig(
                name=name,
                rank=int(r["rank"]),
                coupon=float(r["coupon"]),
                is_residual=bool(r["is_residual"]),
            )
        )
        tranche_states[name] = TrancheState(opening_balance=float(r["opening_balance"]), interest_shortfall=0.0)

    deal = DealConfig(
        day_count_fraction=float(deal_kv["day_count_fraction"]),
        oc_trigger=float(deal_kv["oc_trigger"]),
        ic_trigger=float(deal_kv["ic_trigger"]),
        reserve_target=float(deal_kv["reserve_target"]),
        fees=fees,
    )

    period = PeriodInputs(
        payment_date=str(deal_kv["payment_date"]),
        collateral_balance=float(deal_kv["collateral_balance"]),
        interest_collections=float(deal_kv["interest_collections"]),
        principal_collections=float(deal_kv["principal_collections"]),
        reserve_opening=float(deal_kv["reserve_opening"]),
    )

    return EngineConfig(deal=deal, tranches=tranches), period, tranche_states
