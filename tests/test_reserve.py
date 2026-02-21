from securitisation_engine.models import DealConfig, PeriodInputs, TrancheConfig, TrancheState
from securitisation_engine.config import EngineConfig
from securitisation_engine.waterfall import run_waterfall


def test_reserve_draw_covers_interest_shortfall():
    cfg = EngineConfig(
        deal=DealConfig(
            day_count_fraction=0.25,
            oc_trigger=1.00,
            ic_trigger=1.00,
            reserve_target=50_000,
            fees={},
        ),
        tranches=[
            TrancheConfig("A", 1, 0.08, False),
            TrancheConfig("B", 2, 0.10, False),
            TrancheConfig("C", 3, 0.0, True),
        ],
    )

    period = PeriodInputs(
        payment_date="2026-03-25",
        collateral_balance=12_000_000,
        interest_collections=20_000,     # intentionally low
        principal_collections=0,
        reserve_opening=50_000,          # reserve available
    )

    states = {"A": TrancheState(1_000_000), "B": TrancheState(1_000_000), "C": TrancheState(0)}
    res = run_waterfall(cfg, period, states)

    # A due: 1m*8%*0.25=20k, B due: 1m*10%*0.25=25k => total 45k
    # Interest collections 20k => need 25k reserve draw to fully pay
    assert abs(res.reserve_draw - 25_000) < 1e-6
    assert abs(res.tranche_interest_paid["A"] - 20_000) < 1e-6
    assert abs(res.tranche_interest_paid["B"] - 25_000) < 1e-6
    assert abs(res.reserve_closing - 25_000) < 1e-6
