import math

from securitisation_engine.models import DealConfig, PeriodInputs, TrancheConfig, TrancheState
from securitisation_engine.config import EngineConfig
from securitisation_engine.waterfall import run_waterfall


def test_basic_no_triggers():
    cfg = EngineConfig(
        deal=DealConfig(
            day_count_fraction=0.25,
            oc_trigger=1.05,
            ic_trigger=1.01,
            reserve_target=50000,
            fees={"Servicer": 1000},
        ),
        tranches=[
            TrancheConfig("A", 1, 0.04, False),
            TrancheConfig("B", 2, 0.06, False),
            TrancheConfig("C", 3, 0.0, True),
        ],
    )

    period = PeriodInputs(
        payment_date="2026-03-25",
        collateral_balance=12_000_000,
        interest_collections=200_000,
        principal_collections=300_000,
        reserve_opening=50_000,
    )

    states = {
        "A": TrancheState(8_000_000),
        "B": TrancheState(2_500_000),
        "C": TrancheState(0),
    }

    res = run_waterfall(cfg, period, states)

    a_due = 8_000_000 * 0.04 * 0.25
    b_due = 2_500_000 * 0.06 * 0.25

    assert math.isclose(res.tranche_interest_paid["A"], a_due, rel_tol=1e-9)
    assert math.isclose(res.tranche_interest_paid["B"], b_due, rel_tol=1e-9)
    assert res.tranche_balance_closing["A"] == 8_000_000 - 300_000  # sequential principal all to A first
    assert res.oc_breached is False
    assert res.ic_breached is False
