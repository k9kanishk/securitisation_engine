from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .excel_writer import ensure_template, write_ipd_pack
from .inputs import read_engine_inputs
from .reporting import (
    build_available_funds,
    build_investor_summary,
    build_note_rollforward,
    build_priority_of_payments,
)
from .waterfall import run_waterfall


def run_ipd_engine(
    input_xlsx: str,
    template_xlsx: str,
    output_xlsx: str,
    extra_sheets: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end engine run:
      - Reads engine input workbook (Deal/Fees/Tranches)
      - Runs waterfall
      - Builds IPD output tables
      - Writes the IPD pack into output_xlsx (from template_xlsx)
      - Returns the DataFrames for UI display
    """
    ensure_template(template_xlsx)

    cfg, period, tranche_states = read_engine_inputs(input_xlsx)
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

    if extra_sheets:
        dfs.update(extra_sheets)

    write_ipd_pack(template_xlsx, output_xlsx, dfs)
    return dfs
