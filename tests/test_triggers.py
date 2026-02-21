from securitisation_engine.models import DealConfig, PeriodInputs, TrancheConfig, TrancheState
from securitisation_engine.config import EngineConfig
from securitisation_engine.waterfall import run_waterfall


def test_ic_breach_skips_mezz_interest_and_turbos_senior_principal():
    cfg = EngineConfig(
        deal=DealConfig(
            day_count_fraction=0.25,
            oc_trigger=1.00,
            ic_trigger=1.20,     # hard to pass => breach
            reserve_target=0,
            fees={},
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
        interest_collections=100_000,
        principal_collections=0,
        reserve_opening=0,
    )

    states = {"A": TrancheState(8_000_000), "B": TrancheState(2_500_000), "C": TrancheState(0)}
    res = run_waterfall(cfg, period, states)

    # A interest due = 80k; pay A only, leftover 20k becomes turbo principal to A
    assert abs(res.tranche_interest_paid["A"] - 80_000) < 1e-6
    assert abs(res.tranche_interest_paid["B"] - 0.0) < 1e-6
    assert abs(res.tranche_principal_paid["A"] - 20_000) < 1e-6
    assert res.ic_breached is True


def test_oc_breach_sends_all_principal_to_senior():
    cfg = EngineConfig(
        deal=DealConfig(
            day_count_fraction=0.25,
            oc_trigger=2.00,     # force breach
            ic_trigger=1.00,
            reserve_target=0,
            fees={},
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
        reserve_opening=0,
    )

    states = {"A": TrancheState(8_000_000), "B": TrancheState(2_500_000), "C": TrancheState(0)}
    res = run_waterfall(cfg, period, states)

    assert res.oc_breached is True
    assert abs(res.tranche_principal_paid["A"] - 300_000) < 1e-6
    assert abs(res.tranche_principal_paid["B"] - 0.0) < 1e-6
