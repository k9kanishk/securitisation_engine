from __future__ import annotations

import pandas as pd


def main():
    deal = pd.DataFrame(
        [
            {"key": "payment_date", "value": "2026-03-25"},
            {"key": "day_count_fraction", "value": 0.25},
            {"key": "oc_trigger", "value": 1.10},
            {"key": "ic_trigger", "value": 1.05},
            {"key": "reserve_target", "value": 50000},
            {"key": "reserve_opening", "value": 45000},
            {"key": "collateral_balance", "value": 12000000},
            {"key": "interest_collections", "value": 180000},
            {"key": "principal_collections", "value": 250000},
        ]
    )

    fees = pd.DataFrame(
        [
            {"fee_name": "Servicer Fee", "amount": 5000},
            {"fee_name": "Trustee Fee", "amount": 1000},
        ]
    )

    tranches = pd.DataFrame(
        [
            {"name": "A", "rank": 1, "coupon": 0.045, "opening_balance": 8000000, "is_residual": False},
            {"name": "B", "rank": 2, "coupon": 0.070, "opening_balance": 2500000, "is_residual": False},
            {"name": "C", "rank": 3, "coupon": 0.000, "opening_balance": 0, "is_residual": True},
        ]
    )

    with pd.ExcelWriter("sample_input.xlsx") as xw:
        deal.to_excel(xw, sheet_name="Deal", index=False)
        fees.to_excel(xw, sheet_name="Fees", index=False)
        tranches.to_excel(xw, sheet_name="Tranches", index=False)

    print("Created sample_input.xlsx")


if __name__ == "__main__":
    main()
