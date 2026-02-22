# BMW ABS IPD Reporting Engine (EDGAR → Exhibit 99.1 → Waterfall → IPD Pack + Reconciliation)

End-to-end securitisation **IPD pack generator** built around a real, public ABS deal:
**BMW Vehicle Owner Trust 2025-A** (SEC EDGAR Form 10-D + Exhibit 99.1).

The app:
1) Pulls the latest **Form 10-D** for a BMW trust series from EDGAR
2) Finds and downloads **Exhibit 99.1**
3) Parses **real cashflows, fees, tranche balances, interest/principal distributions**
4) Runs a waterfall/IPD engine
5) Produces an **Excel IPD pack** and a **Reconciliation sheet** proving outputs match Exhibit 99.1 (to cents)

---

## Why this project matters (what it demonstrates)

This replicates what cash management / structured finance ops teams do every payment date:
- Validate **available funds** (interest & principal)
- Apply **priority of payments**
- Roll **note balances**
- Track **reserve account movements**
- Produce an **investor report**
- Reconcile internal calculations vs **trustee/servicer reporting**

The **Reconciliation sheet** is the key deliverable: it shows `Exhibit vs Engine vs Diff` with PASS/FAIL.

---

## Key Features

### ✅ Real Data Ingestion (EDGAR)
- Uses SEC EDGAR `data.sec.gov` submissions + Archives to locate latest 10-D
- Extracts Exhibit 99.1 URL and downloads the statement

### ✅ Robust Exhibit 99.1 Parsing
- Table-based extraction via `pandas.read_html`
- Works even when HTML table headers are missing or inconsistent
- Includes identity-based principal table parsing (Begin − Paid = End)

### ✅ Waterfall / IPD Outputs
- **Available Funds** statement (Interest/Principal sources + reserve)
- **Priority of Payments** (Interest + Principal)
- **Note Rollforward** (Opening, Paid, Closing; Interest due/paid/shortfall)
- **Investor Summary**
- **Reconciliation** (Exhibit vs Engine)

### ✅ BMW “Exact Mode” Logic Implemented
To match Exhibit 99.1 for BMW 2025-A:
- Tranche-specific interest day count fraction inferred from Exhibit payments
  (fixed-rate notes behave like 30/360; floating behaves like ACT/360)
- Principal paid to notes is capped to the Exhibit note principal distributable amount
- Principal allocated **pro-rata within rank** (A-2a and A-2b share rank and split)
- Residual (Certificates) receives leftover principal + residual interest

---

## Project Structure

```text
securitisation_engine/
app.py # Streamlit app entrypoint (UI + downloads)
securitisation_engine/
data_sources/
sec_edgar.py # SEC EDGAR client (submissions + archives)
bmw_owner_trust.py # BMW Exhibit 99.1 parsing + input builder + recon metrics
waterfall.py # Waterfall engine (BMW exact mode support)
reporting.py # Builds output tables (Available Funds, PoP, Rollforward, Summary)
reconciliation.py # Exhibit vs Engine reconciliation builder
runner.py # Orchestrates engine run + Excel pack write
excel_writer.py # Excel template + output writer
inputs.py # Reads engine inputs from Excel
models.py # Deal/Tranche/Period models
config.py # EngineConfig sorting/grouping helpers
```

---

## Requirements

- Python 3.10+ recommended
- Packages are installed via `requirements.txt`

Key deps:
- `streamlit`, `requests`
- `pandas`, `numpy`
- `openpyxl`
- `lxml` (or `bs4/html5lib` depending on your environment)
- `python-dateutil`

---

## Run Locally

### 1) Create venv & install deps

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Set SEC User-Agent (required)

SEC expects a real identifying User-Agent string.

```bash
export SEC_USER_AGENT="YourName your.email@domain.com"
```

### 3) Run app

```bash
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Push repo to GitHub
2. Streamlit Cloud → New app → select repo/branch → main file: `app.py`
3. Set Secrets:

```toml
SEC_USER_AGENT="YourName your.email@domain.com"
```

---

## How to Use the App

### EDGAR Mode

- Enter BMW trust CIK (default: `2049336` for BMW Vehicle Owner Trust 2025-A)
- Click **Run end-to-end**
- Download:
  - Engine input workbook (Deal/Fee/Tranche sheets)
  - IPD Pack XLSX

### Upload Mode

- Upload a compatible engine input workbook
- App runs engine + generates IPD pack

---

## Output: What’s in the IPD Pack

### Available Funds
- Interest collections, principal collections, reserve movements

### Priority of Payments
- Interest waterfall: fees → tranche interest → reserve moves → residual
- Principal waterfall: pro-rata by rank + residual

### Note Rollforward
- Opening balance → principal paid → closing balance
- Interest due/paid + shortfall tracking

### Reconciliation (Most Important Sheet)
- Shows Exhibit 99.1 vs Engine vs Diff vs PASS/FAIL
- Includes:
  - Control totals (Available Interest/Principal/Funds)
  - Distribution totals (fees, note interest, note principal, certificates, total)
  - Reserve opening/closing
  - Tranche-level interest and principal checks

If Reconciliation is all PASS with Diff = 0.00, the run is considered correct.

---

## Notes / Limitations (Transparent + Interview-Safe)

- This implementation matches BMW Vehicle Owner Trust 2025-A Exhibit 99.1 outputs for the tested period(s).
- Other trusts/deals can have materially different mechanics (turbo, traps, OC release rules, step-up coupons, different day count conventions, etc.).
- OC/IC ratios shown are simplified unless explicitly mapped to a deal’s definitions.
- This is designed to be ops-grade validation tooling rather than a full rating agency model.

---

## Testing / Validation

Primary validation is the Reconciliation sheet versus Exhibit 99.1.

Recommended workflow:
1. Run for latest period → ensure PASS everywhere
2. Run for at least one prior month → ensure PASS again
3. If any FAIL appears, investigate parsing changes or deal mechanic changes

---

## Disclaimer

This is for educational / demonstration purposes using publicly available SEC filings. It is not investment advice and not an official trustee/servicer report.
