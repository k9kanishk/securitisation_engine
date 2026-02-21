from __future__ import annotations

from securitisation_engine.excel_writer import ensure_template


def main():
    ensure_template("ipd_template.xlsx")
    print("Created ipd_template.xlsx")


if __name__ == "__main__":
    main()
