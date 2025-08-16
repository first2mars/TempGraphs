# SimpleWeather — Hourly Climatology + Chamber Overlay

## Below Description is of climo_overlay.py

A utility to build **monthly hourly temperature climatologies** from **climate.af.mil** station CSVs (converted to **local time**), and to **overlay a climate‑chamber test profile**. Also supports **composite plots** across many stations.

---

## What it does
- **Monthly climatology per station (choose any month)**
  - Per **local hour** (0–23): **mean, std, min, max, p05, p25, p75, p95**
  - Uses the most recent **N years** present in the file (default **10**)
- **Overlay a chamber test** (`hour,temp`; usually 24 or 48 points)
  - Climatology is **interpolated** to the test hours
- **Composite mode** over a folder of stations
  - Auto‑discovers CSVs whose filenames start with a **four‑letter ID** (e.g., `KDLF_*.csv` or `KDLF-*.csv`)
  - Draws **one mean line per station** and (unless `--average_only`) shades **25–75% (IQR)** and **5–95%** ranges
  - Can **overlay a chamber test** on the composite via `--composite_test`
- **Extreme‑days callouts** (for the selected month)
  - Average **days per year** in these bins (shown only if non‑zero): **100–109°F**, **≥110°F**, **−5 to −9°F**, **≤−10°F**
  - On the charts, you’ll see a label **“Average days per year”** above a small values box

---

## Requirements & setup
**Python 3.9+** recommended.

```
pandas>=2.0.0
numpy>=1.25.0
matplotlib>=3.7.0
tzdata>=2023.3
backports.zoneinfo; python_version < "3.9"
```

Create a venv and install:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Input formats
### Station CSV (from climate.af.mil)
Required columns:
- `Date` in `MM-DD-YYYY`
- `Time (UTC)` in `HH:MM`
- `Air Temp (F)` numeric

> Climatology is computed for your selected **month** using **local time**.

### Chamber test CSV
- Headers not case sensitive; required columns:
  - `hour` → 0, 0.5, 1.0, …, 23.5 (local time)
  - `temp` → °F

---

## Usage

### Single‑station (by explicit file path)
```bash
python climo_overlay.py \
  --raw "./data/KDLF_ICAO_20150101_20250101.csv" \
  --test "./data/111Ftest.csv" \
  --outdir "./outputs" \
  --years 10 \
  --month 8 \
  # --average_only           # optional: mean line only (no bands)
```

### Single‑station (by 4‑letter ID, or multiple IDs)
Finds the latest matching CSV(s) in `--data_dir` and runs single‑station flow for each:
```bash
python climo_overlay.py \
  --station "KDLF,KEND" \
  --data_dir "./data" \
  --test "./data/111Ftest.csv" \
  --outdir "./outputs" \
  --years 10 \
  --month 8 \
  # --average_only
```

### Composite across many stations (auto‑discover)
Scans `--data_dir` for files beginning with a 4‑letter ID:
```bash
python climo_overlay.py \
  --outdir "./outputs" \
  --data_dir "./data" \
  --years 10 \
  --month 8 \
  --composite \
  # --average_only \
  # --stations "KDLF,KEND" \
  # --composite_test "./data/111Ftest.csv"
```

> **Note:** In composite mode you do **not** pass `--raw`. Use `--data_dir` (and optionally `--stations`) instead. For single‑station you can use **either** `--raw` **or** `--station` (list allowed); when both are provided, `--raw` takes priority.

---

## Output structure
```
outputs/
  stations/
    <IDENT>/
      <mon>/
        climo.csv
        <IDENT>_<mon>_overlay.png          # if --test
        <IDENT>_<mon>_residuals.png        # if --test
        merged_test_vs_climo.csv           # if --test

  composite/
    <mon>/
      climo.csv
      composite_<mon>_mean_std_local_<START>_<END>_<N>stns.png
```
- `<IDENT>` is the first 4 characters of the station filename (e.g., `KDLF`).
- `<mon>` is the lowercase month abbreviation (e.g., `aug`).
- `<START>_<END>` is the climatology window (e.g., `2015_2024`).
- Composite PNG includes the month, year span, and **station count**.

---

## What the charts show
### Single‑station overlay
- Mean line (always)
- Bands (unless `--average_only`): **±1 SD**, **IQR (25–75%)**, **5–95%**
- **Chamber Profile** line (from `--test`)
- **Callout** titled **“Average days per year”** with the station’s non‑zero extreme‑day averages
- **Residuals**: bar/line chart of `test − climatology mean` by hour

### Composite
- One **mean line** per station
- Optional shaded ranges (omit with `--average_only`): **25–75% (IQR)** and **5–95% Range**
- **Legend** includes labeled entries for the shaded ranges when drawn
- **Callout** titled **“Average days per year”** listing each station’s non‑zero extreme‑day bins
- Optional **Chamber Profile** overlay (`--composite_test`)

> With `--years 1`, shaded ranges reflect **day‑to‑day** variability within that single month/year. With more years, they blend **year‑to‑year** and **day‑to‑day** variations.

---

## Notes & robustness
- Stats aggregate by **integer local hour** (0–23) to avoid over‑fragmenting bins
- Percentiles use **NaN‑safe** calculations
- CSV reads use explicit dtypes and `low_memory=False`
- Composite auto‑discovery **skips** non‑station files and logs an info message
- Plot layering ensures text is visible: fills < lines < chamber < callout text

---

## Flags (quick reference)
```
--raw             Path to a station CSV (single‑station mode)
--station         Comma‑separated 4‑letter IDs (single‑station mode; uses --data_dir)
--test            Chamber test CSV (hour,temp). Required unless --composite.
--outdir          Output directory root (default: ./outputs)
--data_dir        Directory containing station CSVs (for --station & composite)
--years           Most recent years to average (default: 10)
--month           Month number (1=Jan .. 12=Dec). Default: 8 (Aug)
--tz              IANA timezone for local hour (default: America/Chicago)

--composite       Enable composite mode (scan --data_dir)
--stations        Comma‑separated station IDs to include (composite filter)
--composite_test  Chamber test CSV to overlay on composite
--average_only    Mean line only (hide shaded ranges) — applies to both modes
```

---

## Troubleshooting
- **`KeyError: 'hour'`** — Ensure your test CSV has `hour,temp` and `hour` is numeric (e.g., `0, 0.5, …, 23.5`).
- **Dtype warnings** — We set explicit dtypes; if you still see warnings, check station CSV columns and header row.
- **Empty month** — Confirm your station file contains data for the requested `--month` and within the chosen `--years` window.
- **Composite skipped a file** — Filenames must begin with a **4‑letter ID**; test profiles like `111Ftest.csv` are skipped from discovery by design.

---

## Changelog (recent)
- Added **`--month`** (any month)
- **`--average_only`** in both single‑station and composite
- Composite **auto‑discovers** by four‑letter prefix; supports **`--stations`** filter
- Single‑station supports **`--station`** (ID or comma‑separated list) via `--data_dir`
- Added **extreme‑days callouts** with **“Average days per year”** label
- Cleaned output layout + **descriptive PNG names**
- Test overlay available in **single** (`--test`) and **composite** (`--composite_test`) modes
- Composite shaded regions updated to **25–75% (IQR)** and **5–95% Range**
- Robust dtypes, NaN‑safe percentiles, and layered plotting so annotations stay visible

## Below is a description of the `stats.py` file

Here’s a clean, drop-in README.md for stats.py that you can paste into your repo.

⸻

Temperature vs Boundary Case Study Generator (stats.py)

Generates monthly “case study” reports comparing observed surface temperatures to a certification boundary profile. Computes two risks, writes CSVs/plots, and assembles a polished PDF with clear probability statements.

What it does
	•	Builds 30-min daily temperature profiles in the selected month and years (local time).
	•	Risk 1 (Peak exceedance): daily max temp > boundary peak temp.
	•	Risk 2 (Thermal loading): after peak-aligning boundary per day, any continuous N-hour window with observed > boundary.
	•	Computes 95% Wilson CIs and degree-hours above the boundary.
	•	Saves CSVs, plots (examples, stacked, severity histogram), and a PDF.

Install

# Recommended: a fresh venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# If PDF build fails, add:
pip install reportlab
# On Windows (or minimal distros) for time zones:
pip install tzdata

Data format

Weather CSV (columns):
	•	Date (MM-DD-YYYY)
	•	Time (UTC) (HH:MM)
	•	Air Temp (F)

Boundary CSV (columns):
	•	hour (0, 0.5, …, 23.5)
	•	temp (°F)

Usage

python stats.py \
  --weather /path/KEDW_ICAO_20150101_20250101.csv \
  --boundary /path/111FtestCorrected.csv \
  --station KEDW \
  --month 7 \
  --years 2015-2025 \
  --tz America/Los_Angeles \
  --risk2-hours 2 \
  --outdir ./outputs

Examples

July 2015–2024 (aggregate the month across years):

python stats.py --weather ./data/KEDW_ICAO_20150101_20250101.csv \
  --boundary ./data/111FtestCorrected.csv \
  --station KEDW --month 7 --years 2015-2025 \
  --tz America/Los_Angeles --risk2-hours 2 --outdir ./outputs

Just July 2007:

python stats.py ... --month 7 --years 2007

Multiple months (each month analyzed separately):

python stats.py ... --month 1 --month 7 --years 2015-2020

Change Risk 2 window to 4 hours:

python stats.py ... --month 7 --years 2015-2024 --risk2-hours 4

Outputs

Saved to --outdir with a stem like KEDW_2015-2024-07_*:
	•	*_risk1_daily_peaks.csv
	•	*_risk2_2h.csv (or *_risk2_4h.csv, etc.)
	•	*_risk1_examples.png
	•	*_risk1_stacked.png (with boundary peak line)
	•	*_risk2_examples.png (with per-day shifted boundary in panels)
	•	*_risk2_stacked.png (with one unshifted boundary curve overlay)
	•	*_severity_hist.png
	•	*_CaseStudy.pdf (clear % probabilities + 95% CI)

CLI options (short)

Option	Meaning
--weather	Path to weather CSV
--boundary	Path to boundary CSV
--station	Label in titles/filenames
--month	Month 1–12 (repeatable)
--years	Year(s) or range (e.g., 2007, 2015-2025, 2015,2017,2020)
--tz	IANA zone (e.g., America/Los_Angeles)
--risk2-hours	Window length for Risk 2
--outdir	Output directory
--title	Custom PDF title (optional)

Interpretation
	•	Risk 1: “Estimated probability is X%. With 95% confidence, true probability lies between Y%–Z%.”
	•	Risk 2: Same structure; N-hour window after peak-aligning boundary.

Peak alignment: For Risk 2, the daily boundary curve is circularly shifted so its peak aligns with the observed peak for that day, then the window exceedance is tested.

Notes & assumptions
	•	Time zone conversion uses zoneinfo (install tzdata if needed).
	•	Matplotlib runs headless (Agg backend).
	•	Report generation requires reportlab.

Troubleshooting
	•	Month filter sanity check: the script prints one debug line:

[DEBUG] Filtered records: <rows> | Days: <unique days> | Years: [...] | Month: <m>

If days ≈ 31 × number of years (e.g., ~310 for July 2015–2024), you’re good.

	•	Mismatch warning:

[WARN] Unique dates from resample/group (X) != calendar day count (Y).

Usually harmless (missing/extra partial days). The analysis proceeds using the grouped days.

	•	PDF not created: install reportlab:

pip install reportlab


	•	Time zone issues on Windows: install tzdata:

pip install tzdata



Repro tips
	•	Pin versions via requirements.txt (provided).
	•	Keep boundary grid at 0.5-hour increments for clean interpolation.
	•	For winter analysis, just change --month and keep the same workflow.

⸻

If you want, I can also add a tiny Makefile or bash script to batch run all months and a specific year range into a single outputs/ tree.
