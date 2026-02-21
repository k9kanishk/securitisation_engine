from __future__ import annotations

import argparse

from .inputs import read_engine_inputs
from .waterfall import run_waterfall
from .reporting import (
    build_available_funds,
    build_priority_of_payments,
    build_note_rollforward,
    build_investor_summary,
)
from .excel_writer import ensure_template, write_ipd_pack


def main():
    p = argparse.ArgumentParser(description="Securitisation Waterfall + IPD Reporting Engine")
    p.add_argument("--input", required=True, help="Input Excel file (Deal, Fees, Tranches sheets)")
    p.add_argument("--template", required=True, help="Excel template path (will be created if missing)")
    p.add_argument("--output", required=True, help="Output IPD pack path")
    args = p.parse_args()

    try:
        ensure_template(args.template)
    except Exception:
        # if template already exists and is valid, ensure_template may overwrite; we avoid that by:
        # just try opening it by writer later. If ensure_template fails, continue.
        pass

    cfg, period, tranche_states = read_engine_inputs(args.input)

    tranche_opening = {k: v.opening_balance for k, v in tranche_states.items()}

    res = run_waterfall(cfg, period, tranche_states)

    af = build_available_funds(period, res)
    pop_i, pop_p = build_priority_of_payments(res)
    roll = build_note_rollforward(cfg, res, tranche_opening=tranche_opening)
    inv = build_investor_summary(period, res)

    dfs = {
        "Available Funds": af,
        "Priority of Payments - Interest": pop_i,
        "Priority of Payments - Principal": pop_p,
        "Note Rollforward": roll,
        "Investor Summary": inv,
    }

    write_ipd_pack(args.template, args.output, dfs)
    print(f"Wrote IPD pack: {args.output}")


if __name__ == "__main__":
    main()
