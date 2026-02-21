from __future__ import annotations

from typing import Dict, List, Tuple

from .config import EngineConfig
from .models import PeriodInputs, TrancheState, WaterfallResults


def _safe_div(n: float, d: float) -> float:
    return n / d if d != 0 else float("inf")


def run_waterfall(
    cfg: EngineConfig,
    period: PeriodInputs,
    tranche_states: Dict[str, TrancheState],
) -> WaterfallResults:
    """
    Core logic:
      Interest waterfall:
        1) Fees (from interest)
        2) Reserve draw (if needed) to cover tranche interest shortfalls
        3) Pay interest by rank (unless IC breached: pay A only)
        4) Replenish reserve from remaining interest up to target
        5) Residual interest to residual tranche (if exists), else tracked as excess

      Principal waterfall:
        1) Pay principal sequentially (unless OC breached: all to A)
        2) Any turbo principal from IC breach (from leftover interest) is applied to A first
    """
    tranches_sorted = cfg.sorted_tranches()
    non_residual = [t for t in tranches_sorted if not t.is_residual]
    residual = [t for t in tranches_sorted if t.is_residual]
    residual_name = residual[0].name if residual else None

    dcf = cfg.deal.day_count_fraction

    # Compute interest due (including any carried shortfall)
    interest_due: Dict[str, float] = {}
    for t in non_residual:
        st = tranche_states[t.name]
        due = st.opening_balance * t.coupon * dcf + st.interest_shortfall
        interest_due[t.name] = max(due, 0.0)

    # IC test uses A+B only (typical simplified)
    a_name = non_residual[0].name
    b_name = non_residual[1].name if len(non_residual) > 1 else None
    ab_interest_due = interest_due[a_name] + (interest_due[b_name] if b_name else 0.0)
    ic_ratio = _safe_div(period.interest_collections, ab_interest_due)
    ic_breached = ic_ratio < cfg.deal.ic_trigger

    # OC test uses A+B note balance only
    ab_bal = tranche_states[a_name].opening_balance + (tranche_states[b_name].opening_balance if b_name else 0.0)
    oc_ratio = _safe_div(period.collateral_balance, ab_bal)
    oc_breached = oc_ratio < cfg.deal.oc_trigger

    # Start interest available
    interest_avail = float(period.interest_collections)
    principal_avail = float(period.principal_collections)

    interest_uses: List[Dict[str, float]] = []
    principal_uses: List[Dict[str, float]] = []

    # 1) Fees from interest
    fees_paid_total = 0.0
    for fee_name, amt in cfg.deal.fees.items():
        pay = min(interest_avail, max(float(amt), 0.0))
        interest_avail -= pay
        fees_paid_total += pay
        interest_uses.append({"step": f"Fee: {fee_name}", "amount": pay})

    # 2) Determine reserve draw needed to support interest payments
    reserve_open = float(period.reserve_opening)
    reserve_draw = 0.0

    # If IC breached, we only intend to pay A interest (not B/C).
    intended_interest_names = [a_name] if ic_breached else [t.name for t in non_residual]

    intended_interest_total = sum(interest_due[n] for n in intended_interest_names)

    if interest_avail < intended_interest_total and reserve_open > 0:
        need = intended_interest_total - interest_avail
        reserve_draw = min(reserve_open, need)
        reserve_open -= reserve_draw
        interest_avail += reserve_draw
        interest_uses.append({"step": "Reserve Draw (to cover interest)", "amount": reserve_draw})

    # 3) Pay tranche interest
    interest_paid: Dict[str, float] = {t.name: 0.0 for t in tranches_sorted}
    interest_shortfall_close: Dict[str, float] = {t.name: 0.0 for t in tranches_sorted}

    if ic_breached:
        # Pay A only
        pay = min(interest_avail, interest_due[a_name])
        interest_avail -= pay
        interest_paid[a_name] = pay
        interest_uses.append({"step": f"Tranche {a_name} Interest", "amount": pay})

        # unpaid due becomes shortfall (A only)
        interest_shortfall_close[a_name] = interest_due[a_name] - pay

        # B/C are skipped entirely => full due becomes shortfall
        for t in non_residual[1:]:
            interest_shortfall_close[t.name] = interest_due[t.name]
            interest_uses.append({"step": f"Tranche {t.name} Interest (skipped IC trigger)", "amount": 0.0})
    else:
        for t in non_residual:
            due = interest_due[t.name]
            pay = min(interest_avail, due)
            interest_avail -= pay
            interest_paid[t.name] = pay
            interest_shortfall_close[t.name] = due - pay
            interest_uses.append({"step": f"Tranche {t.name} Interest", "amount": pay})

    # 3b) IC breach turbo: remaining interest is used as principal to pay down A
    turbo_principal_from_interest = 0.0
    if ic_breached and interest_avail > 0:
        turbo_principal_from_interest = interest_avail
        interest_uses.append({"step": f"IC Turbo Principal to {a_name} (from interest)", "amount": turbo_principal_from_interest})
        interest_avail = 0.0

    # 4) Reserve replenish from remaining interest (only if not used up)
    reserve_replenish = 0.0
    reserve_target = float(cfg.deal.reserve_target)
    # note: we already reduced reserve_open by draw; reserve_open now is "post-draw"
    if interest_avail > 0 and reserve_open < reserve_target:
        need = reserve_target - reserve_open
        reserve_replenish = min(interest_avail, need)
        interest_avail -= reserve_replenish
        reserve_open += reserve_replenish
        interest_uses.append({"step": "Reserve Replenishment (from interest)", "amount": reserve_replenish})

    # 5) Residual / excess interest
    excess_interest = interest_avail
    if residual_name:
        interest_paid[residual_name] = excess_interest
        interest_uses.append({"step": f"Residual Interest to {residual_name}", "amount": excess_interest})
    else:
        interest_uses.append({"step": "Excess Interest (unallocated)", "amount": excess_interest})
    interest_avail = 0.0

    # --- Principal waterfall (including turbo from interest if IC breached) ---
    principal_total = principal_avail + turbo_principal_from_interest

    tranche_principal_paid: Dict[str, float] = {t.name: 0.0 for t in tranches_sorted}
    tranche_close: Dict[str, float] = {t.name: tranche_states[t.name].opening_balance for t in tranches_sorted}

    def pay_principal_to(tranche_name: str, amt: float) -> float:
        nonlocal principal_total
        bal = tranche_close[tranche_name]
        pay = min(principal_total, amt, bal)
        principal_total -= pay
        tranche_principal_paid[tranche_name] += pay
        tranche_close[tranche_name] -= pay
        return pay

    if oc_breached:
        # All principal to A until exhausted
        paid = pay_principal_to(a_name, principal_total)
        principal_uses.append({"step": f"OC Trigger: Principal to {a_name}", "amount": paid})
        # others get 0 by definition
        for t in non_residual[1:]:
            principal_uses.append({"step": f"Tranche {t.name} Principal (blocked OC trigger)", "amount": 0.0})
    else:
        # Sequential A -> B -> C (if non-residual)
        for t in non_residual:
            if principal_total <= 0:
                principal_uses.append({"step": f"Tranche {t.name} Principal", "amount": 0.0})
                continue
            paid = pay_principal_to(t.name, principal_total)
            principal_uses.append({"step": f"Tranche {t.name} Principal", "amount": paid})

    # Any remaining principal to residual (if exists), else tracked
    if principal_total > 0:
        if residual_name:
            tranche_principal_paid[residual_name] += principal_total
            principal_uses.append({"step": f"Residual Principal to {residual_name}", "amount": principal_total})
        else:
            principal_uses.append({"step": "Excess Principal (unallocated)", "amount": principal_total})
        principal_total = 0.0

    # Update shortfalls and balances for outputs
    tranche_interest_due_out = {**{k: v for k, v in interest_due.items()}, **({residual_name: 0.0} if residual_name else {})}
    tranche_interest_paid_out = interest_paid

    # Build closing shortfall dict for all tranches
    for t in tranches_sorted:
        if t.is_residual:
            interest_shortfall_close[t.name] = 0.0
        else:
            interest_shortfall_close.setdefault(t.name, 0.0)

    reserve_closing = reserve_open

    return WaterfallResults(
        interest_available=period.interest_collections,
        principal_available=period.principal_collections,
        interest_uses=interest_uses,
        principal_uses=principal_uses,
        reserve_draw=reserve_draw,
        reserve_replenish=reserve_replenish,
        reserve_closing=reserve_closing,
        tranche_interest_due=tranche_interest_due_out,
        tranche_interest_paid=tranche_interest_paid_out,
        tranche_interest_shortfall_closing=interest_shortfall_close,
        tranche_principal_paid=tranche_principal_paid,
        tranche_balance_closing=tranche_close,
        oc_ratio=oc_ratio,
        ic_ratio=ic_ratio,
        oc_breached=oc_breached,
        ic_breached=ic_breached,
        excess_interest_to_residual=excess_interest if residual_name else excess_interest,
    )
