from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .config import EngineConfig
from .models import DealConfig, PeriodInputs, TrancheConfig, TrancheState


def _read_table(path: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def read_engine_inputs(excel_path: str) -> Tuple[EngineConfig, PeriodInputs, Dict[str, TrancheState]]:
    """
    Expected sheets:
      - Deal (key/value table)
      - Fees (fee_name, amount)
      - Tranches (name, rank, coupon, opening_balance, is_residual, [optional: dcf])
    """
    deal_df = _read_table(excel_path, "Deal")
    if set(deal_df.columns) != {"key", "value"}:
        raise ValueError("Deal sheet must have columns: key, value")

    # keep payment_date as string, parse everything else as float when possible
    deal_kv: Dict[str, object] = {}
    for k, v in zip(deal_df["key"], deal_df["value"]):
        ks = str(k)
        if ks == "payment_date":
            deal_kv[ks] = str(v)
        else:
            try:
                deal_kv[ks] = float(v)
            except Exception:
                deal_kv[ks] = v

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

        dcf_val = None
        if "dcf" in tr_df.columns:
            raw = r.get("dcf", None)
            if raw is not None and str(raw).strip() not in ("", "nan", "None"):
                try:
                    dcf_val = float(raw)
                except Exception:
                    dcf_val = None

        tranches.append(
            TrancheConfig(
                name=name,
                rank=int(r["rank"]),
                coupon=float(r["coupon"]),
                is_residual=bool(r["is_residual"]),
                dcf=dcf_val,
            )
        )
        tranche_states[name] = TrancheState(opening_balance=float(r["opening_balance"]), interest_shortfall=0.0)

    principal_cap = None
    if "principal_to_notes_cap" in deal_kv:
        try:
            principal_cap = float(deal_kv["principal_to_notes_cap"])  # type: ignore[arg-type]
        except Exception:
            principal_cap = None

    deal = DealConfig(
        day_count_fraction=float(deal_kv["day_count_fraction"]),   # type: ignore[arg-type]
        oc_trigger=float(deal_kv["oc_trigger"]),                   # type: ignore[arg-type]
        ic_trigger=float(deal_kv["ic_trigger"]),                   # type: ignore[arg-type]
        reserve_target=float(deal_kv["reserve_target"]),           # type: ignore[arg-type]
        fees=fees,
        principal_to_notes_cap=principal_cap,
    )

    period = PeriodInputs(
        payment_date=str(deal_kv["payment_date"]),
        collateral_balance=float(deal_kv["collateral_balance"]),       # type: ignore[arg-type]
        interest_collections=float(deal_kv["interest_collections"]),   # type: ignore[arg-type]
        principal_collections=float(deal_kv["principal_collections"]), # type: ignore[arg-type]
        reserve_opening=float(deal_kv["reserve_opening"]),             # type: ignore[arg-type]
    )

    return EngineConfig(deal=deal, tranches=tranches), period, tranche_states
