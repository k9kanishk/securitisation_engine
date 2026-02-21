from __future__ import annotations

import argparse
import os

from securitisation_engine.data_sources.bmw_owner_trust import fetch_latest_bmw_exhibit991_to_input_xlsx


def main():
    p = argparse.ArgumentParser(description="Fetch latest BMW Vehicle Owner Trust Exhibit 99.1 and build engine input Excel.")
    p.add_argument("--cik", default="2049336", help="CIK of BMW trust series (default: 2049336 = BMW Vehicle Owner Trust 2025-A)")
    p.add_argument("--out", default="bmw_input.xlsx", help="Output Excel path for engine input")
    p.add_argument(
        "--user-agent",
        default=os.getenv("SEC_USER_AGENT", ""),
        help="SEC User-Agent header, e.g. 'YourName your.email@domain.com' (or set SEC_USER_AGENT env var)",
    )
    args = p.parse_args()

    ten_d_url, ex99_url = fetch_latest_bmw_exhibit991_to_input_xlsx(
        cik=args.cik, user_agent=args.user_agent, out_xlsx=args.out
    )

    print(f"Wrote engine input: {args.out}")
    print(f"10-D URL: {ten_d_url}")
    print(f"Exhibit 99.1 URL: {ex99_url}")


if __name__ == "__main__":
    main()
