from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from securitisation_engine.data_sources.bmw_owner_trust import (
    fetch_latest_bmw_exhibit991_to_input_xlsx,
)
from securitisation_engine.runner import run_ipd_engine


st.set_page_config(page_title="BMW ABS IPD Engine", layout="wide")


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_bmw_input(cik: str, user_agent: str) -> Tuple[str, str, bytes]:
    """Fetch latest 10-D and Exhibit 99.1 and return engine input workbook bytes."""
    with tempfile.TemporaryDirectory() as td:
        out_xlsx = str(Path(td) / "bmw_input.xlsx")
        ten_d_url, ex99_url = fetch_latest_bmw_exhibit991_to_input_xlsx(
            cik=cik, user_agent=user_agent, out_xlsx=out_xlsx
        )
        return ten_d_url, ex99_url, _read_bytes(out_xlsx)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_run_engine(input_excel_bytes: bytes) -> Tuple[bytes, Dict[str, pd.DataFrame]]:
    """Run engine and return IPD pack bytes + DataFrames."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        input_xlsx = td / "input.xlsx"
        template_xlsx = td / "ipd_template.xlsx"
        output_xlsx = td / "ipd_pack.xlsx"

        input_xlsx.write_bytes(input_excel_bytes)
        dfs = run_ipd_engine(
            input_xlsx=str(input_xlsx),
            template_xlsx=str(template_xlsx),
            output_xlsx=str(output_xlsx),
        )
        return output_xlsx.read_bytes(), dfs


def main():
    st.title("BMW Vehicle Owner Trust — IPD Pack Generator (Streamlit)")

    with st.sidebar:
        st.header("Mode")

        mode = st.radio("Choose input source", ["EDGAR (latest BMW)", "Upload Excel"], index=0)

        st.divider()
        st.header("EDGAR Settings")

        cik = st.text_input("BMW CIK", value="2049336", help="Digits only. Example: 2049336")
        user_agent = st.text_input(
            "SEC User-Agent (required)",
            value=os.getenv("SEC_USER_AGENT", ""),
            type="password",
            help="Must be real, e.g. 'YourName your.email@domain.com'.",
        )

        run_btn = st.button("Run end-to-end")

        st.divider()
        st.caption("Tip: EDGAR calls are cached for 1 hour to avoid rerun spam.")

    st.write(
        "This app fetches the latest BMW **10-D** and **Exhibit 99.1** from EDGAR (or uses your uploaded workbook), "
        "builds engine inputs, runs the waterfall, and outputs an **IPD XLSX pack**."
    )

    input_bytes = None
    ten_d_url = None
    ex99_url = None

    if mode == "Upload Excel":
        uploaded = st.file_uploader("Upload engine input workbook (Deal / Fees / Tranches)", type=["xlsx"])
        if uploaded is not None:
            input_bytes = uploaded.read()
            st.success("Uploaded input workbook loaded.")
        else:
            st.info("Upload an input workbook to run.")
            return

    else:
        if not run_btn:
            st.info("Click **Run end-to-end** to fetch from EDGAR and generate the pack.")
            return

        if not user_agent.strip():
            st.error("SEC User-Agent is required. Put it in the sidebar (or set SEC_USER_AGENT).")
            return

        if not cik.strip().isdigit():
            st.error("CIK must be digits only.")
            return

        with st.spinner("Fetching latest 10-D + Exhibit 99.1 from EDGAR..."):
            try:
                ten_d_url, ex99_url, input_bytes = cached_fetch_bmw_input(
                    cik=cik.strip(), user_agent=user_agent.strip()
                )
            except Exception as e:
                st.exception(e)
                return

    with st.spinner("Running waterfall + building IPD pack..."):
        try:
            ipd_bytes, dfs = cached_run_engine(input_bytes)
        except Exception as e:
            st.exception(e)
            return

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.download_button(
            "Download engine input (XLSX)",
            data=input_bytes,
            file_name="bmw_engine_input.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        st.download_button(
            "Download IPD pack (XLSX)",
            data=ipd_bytes,
            file_name="bmw_ipd_pack.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col3:
        if ten_d_url and ex99_url:
            st.markdown("**EDGAR links**")
            st.markdown(f"- 10-D: {ten_d_url}")
            st.markdown(f"- Exhibit 99.1: {ex99_url}")

    st.divider()

    tabs = st.tabs(list(dfs.keys()))
    for name, tab in zip(dfs.keys(), tabs):
        with tab:
            st.subheader(name)
            st.dataframe(dfs[name], use_container_width=True)


if __name__ == "__main__":
    main()
