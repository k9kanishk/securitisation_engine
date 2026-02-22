from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .config import EngineConfig
from .models import PeriodInputs, TrancheState, WaterfallResults


def _safe_div(n: float, d: float) -> float:
    return n / d if d != 0 else float("inf")


def _r2(x: float) -> float:
    # stable 2dp rounding to avoid float artifacts
    return float(round((x or 0.0) + 1e-9, 2))


def _alloc_pro_rata(names: List[str], total: float, balances: Dict[str, float]) -> Dict[str, float]:
    """
    Allocate 'total' across 'names' pro-rata by balances, rounded to cents.
    """
    if total <= 0:
        return {n: 0.0 for n in names}

    bals = {n: max(float(balances.get(n, 0.0)), 0.0) for n in names}
    denom = sum(bals.values())

    if denom <= 0:
        # even split fallback
        per = round(total / len(names), 2)
        alloc = {n: per for n in names}
        alloc[names[-1]] = round(total - sum(alloc[n] for n in names[:-1]), 2)
        return alloc

    alloc: Dict[str, float] = {}
    remaining = round(total, 2)

    for n in names[:-1]:
        share = bals[n] / denom
        amt = round(total * share, 2)
        amt = min(amt, bals[n], remaining)
        alloc[n] = amt
        remaining = round(remaining - amt, 2)

    last = names[-1]
    alloc[last] = min(remaining, bals[last])
    diff = round(total - sum(alloc.values()), 2)
    if abs(diff) >= 0.01:
        alloc[last] = round(alloc[last] + diff, 2)

    return alloc


def run_waterfall(
    cfg: EngineConfig,
    period: PeriodInputs,
    tranche_states: Dict[str, TrancheState],
) -> WaterfallResults:
    """
    Interest waterfall:
      1) Fees (from interest)
      2) Reserve draw (if needed) to cover tranche interest shortfalls
      3) Pay interest by rank (unless IC breached: pay A only)
      4) Replenish reserve from remaining interest up to target
      5) Residual interest to residual tranche (if exists), else tracked as excess

    Principal waterfall:
      - BMW-mode support:
        * Optional principal cap to notes (cfg.deal.principal_to_notes_cap)
        * Pro-rata principal allocation within same rank (e.g., A-2a and A-2b share rank=2)
    """
    tranches_sorted = cfg.sorted_tranches()
    non_residual = [t for t in tranches_sorted if not t.is_residual]
    residual = [t for t in tranches_sorted if t.is_residual]
    residual_name = residual[0].name if residual else None

    base_dcf = cfg.deal.day_count_fraction

    # Compute interest due using tranche-specific dcf if provided
    interest_due: Dict[str, float] = {}
    for t in non_residual:
        st = tranche_states[t.name]
        tr_dcf = t.dcf if t.dcf is not None else base_dcf
        due = st.opening_balance * t.coupon * tr_dcf + st.interest_shortfall
        interest_due[t.name] = _r2(max(due, 0.0))

    # IC test uses first two tranches in the sorted set (still "simplified")
    a_name = non_residual[0].name
    b_name = non_residual[1].name if len(non_residual) > 1 else None
    ab_interest_due = interest_due[a_name] + (interest_due[b_name] if b_name else 0.0)
    ic_ratio = _safe_div(period.interest_collections, ab_interest_due)
    ic_breached = ic_ratio < cfg.deal.ic_trigger

    # OC test uses A+B note balance only (simplified)
    ab_bal = tranche_states[a_name].opening_balance + (tranche_states[b_name].opening_balance if b_name else 0.0)
    oc_ratio = _safe_div(period.collateral_balance, ab_bal)
    oc_breached = oc_ratio < cfg.deal.oc_trigger

    interest_avail = float(period.interest_collections)
    principal_avail = float(period.principal_collections)

    interest_uses: List[Dict[str, float]] = []
    principal_uses: List[Dict[str, float]] = []

    # 1) Fees from interest
    for fee_name, amt in cfg.deal.fees.items():
        pay = _r2(min(interest_avail, max(float(amt), 0.0)))
        interest_avail = _r2(interest_avail - pay)
        interest_uses.append({"step": f"Fee: {fee_name}", "amount": pay})

    # 2) Reserve draw to cover intended interest
    reserve_open = float(period.reserve_opening)
    reserve_draw = 0.0

    intended_interest_names = [a_name] if ic_breached else [t.name for t in non_residual]
    intended_interest_total = sum(interest_due[n] for n in intended_interest_names)

    if interest_avail < intended_interest_total and reserve_open > 0:
        need = intended_interest_total - interest_avail
        reserve_draw = _r2(min(reserve_open, need))
        reserve_open = _r2(reserve_open - reserve_draw)
        interest_avail = _r2(interest_avail + reserve_draw)
        interest_uses.append({"step": "Reserve Draw (to cover interest)", "amount": reserve_draw})

    # 3) Pay tranche interest
    interest_paid: Dict[str, float] = {t.name: 0.0 for t in tranches_sorted}
    interest_shortfall_close: Dict[str, float] = {t.name: 0.0 for t in tranches_sorted}

    if ic_breached:
        pay = _r2(min(interest_avail, interest_due[a_name]))
        interest_avail = _r2(interest_avail - pay)
        interest_paid[a_name] = pay
        interest_uses.append({"step": f"Tranche {a_name} Interest", "amount": pay})
        interest_shortfall_close[a_name] = _r2(interest_due[a_name] - pay)

        for t in non_residual[1:]:
            interest_shortfall_close[t.name] = interest_due[t.name]
            interest_uses.append({"step": f"Tranche {t.name} Interest (skipped IC trigger)", "amount": 0.0})
    else:
        for t in non_residual:
            due = interest_due[t.name]
            pay = _r2(min(interest_avail, due))
            interest_avail = _r2(interest_avail - pay)
            interest_paid[t.name] = pay
            interest_shortfall_close[t.name] = _r2(due - pay)
            interest_uses.append({"step": f"Tranche {t.name} Interest", "amount": pay})

    # 3b) IC breach turbo
    turbo_principal_from_interest = 0.0
    if ic_breached and interest_avail > 0:
        turbo_principal_from_interest = interest_avail
        interest_uses.append({"step": f"IC Turbo Principal to {a_name} (from interest)", "amount": turbo_principal_from_interest})
        interest_avail = 0.0

    # 4) Reserve replenish
    reserve_replenish = 0.0
    reserve_target = float(cfg.deal.reserve_target)
    if interest_avail > 0 and reserve_open < reserve_target:
        need = reserve_target - reserve_open
        reserve_replenish = _r2(min(interest_avail, need))
        interest_avail = _r2(interest_avail - reserve_replenish)
        reserve_open = _r2(reserve_open + reserve_replenish)
        interest_uses.append({"step": "Reserve Replenishment (from interest)", "amount": reserve_replenish})

    # 5) Residual interest
    excess_interest = _r2(interest_avail)
    if residual_name:
        interest_paid[residual_name] = excess_interest
        interest_uses.append({"step": f"Residual Interest to {residual_name}", "amount": excess_interest})
    else:
        interest_uses.append({"step": "Excess Interest (unallocated)", "amount": excess_interest})
    interest_avail = 0.0

    # --- Principal waterfall ---
    principal_total_all = principal_avail + turbo_principal_from_interest

    # Apply optional cap to notes; leftover goes to residual
    cap = cfg.deal.principal_to_notes_cap
    if cap is not None and cap >= 0:
        principal_to_notes = min(principal_total_all, cap)
        principal_leftover = principal_total_all - principal_to_notes
    else:
        principal_to_notes = principal_total_all
        principal_leftover = 0.0

    principal_total = principal_to_notes

    tranche_principal_paid: Dict[str, float] = {t.name: 0.0 for t in tranches_sorted}
    tranche_close: Dict[str, float] = {t.name: tranche_states[t.name].opening_balance for t in tranches_sorted}

    def pay_principal_to(tranche_name: str, amt: float) -> float:
        nonlocal principal_total
        bal = tranche_close[tranche_name]
        pay = _r2(min(principal_total, amt, bal))
        principal_total = _r2(principal_total - pay)
        tranche_principal_paid[tranche_name] = _r2(tranche_principal_paid[tranche_name] + pay)
        tranche_close[tranche_name] = _r2(tranche_close[tranche_name] - pay)
        return pay

    if oc_breached:
        paid = pay_principal_to(a_name, principal_total)
        principal_uses.append({"step": f"OC Trigger: Principal to {a_name}", "amount": paid})
        for t in non_residual[1:]:
            principal_uses.append({"step": f"Tranche {t.name} Principal (blocked OC trigger)", "amount": 0.0})
    else:
        # group by rank, allocate pro-rata within same rank (A-2a & A-2b)
        by_rank: Dict[int, List[str]] = defaultdict(list)
        for t in non_residual:
            by_rank[int(t.rank)].append(t.name)

        for r in sorted(by_rank.keys()):
            names = sorted(by_rank[r])
            if principal_total <= 0:
                for n in names:
                    principal_uses.append({"step": f"Tranche {n} Principal", "amount": 0.0})
                continue

            group_bal = sum(max(tranche_close[n], 0.0) for n in names)
            if group_bal <= 0:
                for n in names:
                    principal_uses.append({"step": f"Tranche {n} Principal", "amount": 0.0})
                continue

            group_pay_total = min(principal_total, group_bal)

            if len(names) == 1:
                paid = pay_principal_to(names[0], group_pay_total)
                principal_uses.append({"step": f"Tranche {names[0]} Principal", "amount": paid})
                continue

            alloc = _alloc_pro_rata(names, group_pay_total, tranche_close)
            for n in names:
                if principal_total <= 0:
                    principal_uses.append({"step": f"Tranche {n} Principal", "amount": 0.0})
                    continue
                pay = pay_principal_to(n, alloc[n])
                principal_uses.append({"step": f"Tranche {n} Principal", "amount": pay})

    # Residual principal = (unallocated notes principal) + (cap leftover)
    residual_principal_amt = _r2(principal_total + principal_leftover)
    if residual_principal_amt > 0:
        if residual_name:
            tranche_principal_paid[residual_name] += residual_principal_amt
            principal_uses.append({"step": f"Residual Principal to {residual_name}", "amount": residual_principal_amt})
        else:
            principal_uses.append({"step": "Excess Principal (unallocated)", "amount": residual_principal_amt})
        principal_total = 0.0

    tranche_interest_due_out = {**{k: v for k, v in interest_due.items()}, **({residual_name: 0.0} if residual_name else {})}
    tranche_interest_paid_out = interest_paid

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
        excess_interest_to_residual=excess_interest,
    )
