Got it — since I can’t directly create the file for you right now, here’s the full README.md content you can copy into a file at the root of your project:

⸻


# SimpleWeather — Hourly Climatology + Chamber Overlay

A small utility to build **hourly temperature climatologies** (by month; currently August) from **climate.af.mil** station CSVs, convert to **local time**, and optionally **overlay a chamber test profile**. It also supports **composite plots** across many stations at once.

## Features

- Build **August hourly climatology** from station CSV  
  - Stats per local hour: **mean, std, min, max, 5th, 25th, 75th, 95th percentiles**
  - Uses most recent **N years** present in your file (default **10**)
- **Overlay** a chamber test profile (24 or 48 points, `hour,temp`)
- **Composite mode** across a directory of stations  
  - One plot with a **line per station** (+ optional ±1 SD bands)  
  - Optional **test profile** overlay on the composite
- Organized outputs:  
  - Per station → `./outputs/<IDENT>/...`  
  - Composite → `./outputs/composite/...`
- Station identifier is inferred from the filename prefix (e.g., `KDLF_...csv` → `KDLF`)
- Time zone handled via `zoneinfo` (`America/Chicago` default)

---

## Requirements & Setup

Python 3.9+ recommended.

pandas>=2.0.0
numpy>=1.25.0
matplotlib>=3.7.0
tzdata>=2023.3
backports.zoneinfo; python_version < “3.9”

Create a venv and install:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt


⸻

Data Inputs

Station CSV (climate.af.mil export)

Required columns:
	•	Date in MM-DD-YYYY
	•	Time (UTC) in HH:MM
	•	Air Temp (F) (numeric)

The script builds August climatology using the last N years found in the file.

Chamber test CSV
	•	Columns:
	•	hour → 0, 0.5, 1.0, …, 23.5 (local time)
	•	temp → °F

⸻

Usage

Single-station overlay (build climo + overlay test)

python climo_overlay.py \
  --raw "./data/KDLF_ICAO_20150101_20250101.csv" \
  --test "./data/111Ftest.csv" \
  --outdir "./outputs" \
  --years 10

Outputs (example):

./outputs/KDLF/aug_climatology_local_2015_2024.csv
./outputs/KDLF/overlay_chamber_vs_climo.png
./outputs/KDLF/residuals.png
./outputs/KDLF/merged_test_vs_climo.csv


⸻

Composite across many stations

python climo_overlay.py \
  --raw "./data/KDLF_ICAO_20150101_20250101.csv" \
  --outdir "./outputs" \
  --data_dir "./data" \
  --years 10 \
  --composite

Composite options
	•	Overlay a chamber test on the composite:

--composite_test "./data/111Ftest.csv"


	•	Lines only (no ±1 SD bands):

--average_only


	•	Filter by station identifiers:

--stations "KDLF,KEND"



⸻

CLI Reference

--raw             Path to a station CSV (used to infer defaults and TZ)
--test            Path to chamber test CSV (hour,temp). Required unless --composite.
--outdir          Output directory root (default: ./outputs)
--years           Most recent years to average (default: 10)
--tz              IANA timezone for local hour (default: America/Chicago)

--composite       Scan --data_dir for station CSVs and make a composite plot
--data_dir        Directory to scan for station CSVs
--composite_test  Chamber test CSV to overlay on composite
--average_only    Composite: draw mean lines only (hide ±1 SD bands)
--stations        Comma-separated station identifiers (e.g., KDLF,KEND)


⸻

Output Structure

outputs/
  <IDENT>/
    aug_climatology_local_<START>_<END>.csv
    overlay_chamber_vs_climo.png
    residuals.png
    merged_test_vs_climo.csv

  composite/
    composite_climo_local_<START>_<END>.csv
    composite_mean_std_local_<START>_<END>.png


⸻

Troubleshooting
	•	KeyError: 'hour': Check your test CSV has hour,temp headers.
	•	[WARN] Skipping ...: 'Date': That’s your test CSV; it’s normal to skip in composite mode.
	•	Empty August data: Ensure your station file covers August and has required columns.
	•	Wrong timezone: Use --tz "America/Denver" or similar.

⸻

Roadmap
	•	--month switch instead of fixed August
	•	--lines_only for single-station overlays
	•	Multi-month composites

---

# SimpleWeather — Hourly Climatology + Chamber Overlay

A small utility to build **hourly temperature climatologies** from **climate.af.mil** station CSVs, convert to **local time**, and optionally **overlay a chamber test profile**. It also supports **composite plots** across many stations at once.

---

## What it does

- Build **monthly** (you choose the month) **hourly climatology** per station:
  - Stats for each local hour (0–23): **mean, std, min, max, 5th, 25th, 75th, 95th percentiles**
  - Uses the most recent **N years** present in your file (default **10**)
- **Overlay a chamber test** (24 or 48 points: `hour,temp`), with interpolation of the climatology to your test hours
- **Composite mode**: scan a directory for **many stations** and plot them together
  - One **mean line per station**, optional **±1 SD band** per station
  - Optional **test profile overlay** on the composite
- **Extreme-days callout** (per selected month): average number of days per year in these buckets, shown only if non‑zero
  - **100–109°F**, **≥110°F**, **−5 to −9°F**, **≤−10°F**

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
- `Air Temp (F)` (numeric)

> The script builds climatology for the **month you select** using the last **N years** found in the file. Time conversion uses **local time** (configurable timezone).

### Chamber test CSV
- Columns (headers not case sensitive):
  - `hour` → 0, 0.5, 1.0, …, 23.5 (local time)
  - `temp` → temperature in °F

---

## Usage

### Single-station overlay
Build a monthly climatology for one station and overlay a chamber test profile.

```bash
python climo_overlay.py \
  --raw "./data/KDLF_ICAO_20150101_20250101.csv" \
  --test "./data/111Ftest.csv" \
  --outdir "./outputs" \
  --years 10 \
  --month 8                # 1=Jan .. 12=Dec
  # --average_only         # optional: show only the mean line (no bands)
```

**Outputs (example):**
```
./outputs/KDLF/aug_climatology_local_2015_2024.csv
./outputs/KDLF/overlay_chamber_vs_climo.png
./outputs/KDLF/residuals.png
./outputs/KDLF/merged_test_vs_climo.csv
```
- The **overlay** plot shows the mean line and, unless `--average_only` is set, the **5–95%**, **IQR (25–75%)**, and **±1 SD** bands.
- A callout box lists average **extreme days** (only categories that are non‑zero for that station/month).
- The **residuals** plot shows `test − climatology mean` by hour.

### Composite across many stations
Scan a directory and auto‑discover station CSVs whose filenames **start with a four‑letter identifier** (e.g., `KDLF_*.csv` or `KDLF-*.csv`).

```bash
python climo_overlay.py \
  --outdir "./outputs" \
  --data_dir "./data" \
  --years 10 \
  --month 8 \
  --composite \
  # --average_only             # optional: lines only (no ±1 SD bands)
  # --stations "KDLF,KEND"     # optional: limit to selected identifiers
  # --composite_test "./data/111Ftest.csv"  # optional: overlay chamber test on the composite
```

**Outputs (example):**
```
./outputs/KCBM/aug_climatology_local_2015_2024.csv
./outputs/KDLF/aug_climatology_local_2015_2024.csv
./outputs/KEND/aug_climatology_local_2015_2024.csv
./outputs/KRND/aug_climatology_local_2015_2024.csv

./outputs/composite/composite_aug_climo_local_2015_2024.csv
./outputs/composite/composite_aug_mean_std_local_2015_2024.png
```
- The composite plot shows **one mean line per station**; add bands by omitting `--average_only`.
- A callout box summarizes **extreme‑day averages per station** (only non‑zero categories shown).
- Use `--stations` to restrict to a subset (matches the **first four characters** of filenames).

---

## Command reference

```
--raw             Path to a station CSV (single‑station mode)
--test            Path to chamber test CSV (hour,temp). Required unless --composite.
--outdir          Output directory root (default: ./outputs)
--years           Most recent years to average (default: 10)
--month           Month number to build climatology for (1=Jan .. 12=Dec). Default: 8 (Aug)
--tz              IANA timezone for local hour (default: America/Chicago)

--composite       Scan --data_dir for station CSVs (filenames start with a four‑letter ID) and plot them
--data_dir        Directory to scan for station CSVs
--composite_test  Chamber test CSV to overlay on the composite plot (hour,temp)
--average_only    Draw mean lines only (hide ±1 SD bands) — applies to single‑station and composite
--stations        Comma‑separated station IDs to include (e.g., KDLF,KEND)
```

---

## Output structure

```
outputs/
  <IDENT>/
    <mon>_climatology_local_<START>_<END>.csv
    overlay_chamber_vs_climo.png          # single‑station only
    residuals.png                         # single‑station only
    merged_test_vs_climo.csv              # single‑station only

  composite/
    composite_<mon>_climo_local_<START>_<END>.csv
    composite_<mon>_mean_std_local_<START>_<END>.png
```
- `<IDENT>` is inferred from the filename prefix (e.g., `KDLF_...csv` → `KDLF`).
- `<mon>` is the **lowercase month abbreviation** (e.g., `aug`).
- `<START>_<END>` reflects the climatology window (e.g., `2015_2024`).

---

## Implementation notes

- **Local time:** All stats are aggregated by **local hour**. Set timezone with `--tz` (default `America/Chicago`).
- **Interpolation:** For overlays, climatology is linearly interpolated to your test hours.
- **Robust stats:** Percentiles use **NaN‑safe** calculations; temperature is coerced to numeric.
- **Station detection (composite):** Only files whose names start with a **four‑letter ID** are processed; others are skipped with an info message.

---

## Troubleshooting

- `KeyError: 'hour'` — Ensure your test CSV has columns `hour,temp` and that hours are numeric (e.g., `0, 0.5, …, 23.5`).
- `DtypeWarning` on read — We already set `low_memory=False` and explicit dtypes; if you still see one, verify the station CSV has the required columns and no corrupted header rows.
- Empty month — Confirm your station file contains rows for the requested `--month` and within the chosen `--years` window.
- Composite skipped a file — The filename must start with a **four‑letter ID**; test profiles like `111Ftest.csv` will be skipped automatically.

---

## Changelog (recent)

- Added **`--month`** to support any month (not just August)
- Added **`--average_only`** for both single‑station and composite modes
- Composite mode now **auto‑discovers** stations by **four‑letter filename prefix** (no `--raw` needed)
- Added **extreme‑days callout** to both overlay and composite plots
- Improved dtype handling, robust percentiles, and hour binning