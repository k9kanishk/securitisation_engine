from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TrancheConfig:
    name: str                 # "A", "B", "C"
    rank: int                 # 1=senior, 2=mezz, 3=junior
    coupon: float             # annual coupon, e.g. 0.05
    is_residual: bool = False # True for equity-like residual tranche


@dataclass
class TrancheState:
    opening_balance: float
    interest_shortfall: float = 0.0  # carried unpaid interest


@dataclass(frozen=True)
class DealConfig:
    day_count_fraction: float         # e.g. 0.25 for quarterly
    oc_trigger: float                 # e.g. 1.10
    ic_trigger: float                 # e.g. 1.05
    reserve_target: float             # absolute target balance
    fees: Dict[str, float]            # e.g. {"Servicer Fee": 2000, "Trustee Fee": 500}


@dataclass(frozen=True)
class PeriodInputs:
    payment_date: str                 # string for reporting (YYYY-MM-DD)
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
