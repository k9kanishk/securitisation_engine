from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TrancheConfig:
    name: str
    rank: int
    coupon: float
    is_residual: bool = False
    dcf: Optional[float] = None  # NEW: tranche-specific day count fraction


@dataclass
class TrancheState:
    opening_balance: float
    interest_shortfall: float = 0.0


@dataclass(frozen=True)
class DealConfig:
    day_count_fraction: float
    oc_trigger: float
    ic_trigger: float
    reserve_target: float
    fees: Dict[str, float]
    principal_to_notes_cap: Optional[float] = None  # NEW


@dataclass(frozen=True)
class PeriodInputs:
    payment_date: str
    collateral_balance: float
    interest_collections: float
    principal_collections: float
    reserve_opening: float


@dataclass
class WaterfallResults:
    # Cash movements (each positive number is a payment/use)
    interest_available: float
    principal_available: float

    interest_uses: List[Dict[str, float]]  # ordered steps
    principal_uses: List[Dict[str, float]] # ordered steps

    reserve_draw: float
    reserve_replenish: float
    reserve_closing: float

    tranche_interest_due: Dict[str, float]
    tranche_interest_paid: Dict[str, float]
    tranche_interest_shortfall_closing: Dict[str, float]

    tranche_principal_paid: Dict[str, float]
    tranche_balance_closing: Dict[str, float]

    oc_ratio: float
    ic_ratio: float
    oc_breached: bool
    ic_breached: bool
    excess_interest_to_residual: float
