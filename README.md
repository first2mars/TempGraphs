# SimpleWeather — Hourly Climatology & Chamber Overlay

Utilities for building hourly temperature climatologies from `climate.af.mil` station data, overlaying climate‑chamber test profiles, and evaluating exceedance risk.

The repository offers command-line scripts and a simple GUI:

- `climo_overlay.py` – create monthly hourly temperature climatologies and optionally overlay a chamber‑test profile. Supports single‑station and composite plots.
- `stats.py` – quantify exceedance risk relative to a boundary profile and generate plots and PDF summaries.
- `gui_app.py` – Tkinter interface for running case studies via the `stats.py` options.

## Installation

Python **3.9+** is recommended. Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional extras
pip install reportlab tzdata
```

If you intend to launch the GUI, ensure the standard-library **tkinter** module is available (e.g., install `python3-tk` on Linux).

Key packages (also listed in `requirements.txt`):

```
pandas>=2.0.0
numpy>=1.25.0
matplotlib>=3.7.0
tzdata>=2023.3
backports.zoneinfo; python_version < "3.9"
```

---

## `climo_overlay.py`

### Features

- Build monthly climatologies for any station and month using the most recent *N* years in the file (default **10**).
- Statistics per local hour (0–23): mean, std, min, max, p05, p25, p75, p95.
- Overlay a climate‑chamber test profile; climatology is interpolated to test hours.
- Composite mode to aggregate many stations with interquartile and 5–95% bands.
- Extreme‑day callouts (100–109 °F, ≥110 °F, −5 to −9 °F, ≤−10 °F).

### Input formats

**Station CSV**

Required columns:

- `Date` (MM‑DD‑YYYY)
- `Time (UTC)` (HH:MM)
- `Air Temp (F)`

**Chamber test CSV**

Headers are case‑insensitive; required columns:

- `hour` → 0, 0.5, 1.0, …, 23.5 (local time)
- `temp` → °F

### Usage

Single‑station via explicit file path:

```bash
python climo_overlay.py \
  --raw ./data/KDLF_ICAO_20150101_20250101.csv \
  --test ./data/111Ftest.csv \
  --outdir ./outputs \
  --years 10 \
  --month 8 \
  # --average_only
```

Single‑station via station ID(s):

```bash
python climo_overlay.py \
  --station KDLF,KEND \
  --data_dir ./data \
  --test ./data/111Ftest.csv \
  --outdir ./outputs \
  --years 10 \
  --month 8
```

Composite across many stations (auto‑discover):

```bash
python climo_overlay.py \
  --outdir ./outputs \
  --data_dir ./data \
  --years 10 \
  --month 8 \
  --composite \
  # --average_only \
  # --stations KDLF,KEND \
  # --composite_test ./data/111Ftest.csv
```

> In composite mode do **not** pass `--raw`. For single‑station runs you may use either `--raw` or `--station`; when both are supplied, `--raw` takes priority.

### Output structure

```
outputs/
  stations/
    <IDENT>/<mon>/
      climo.csv
      <IDENT>_<mon>_overlay.png        # if --test
      <IDENT>_<mon>_residuals.png      # if --test
      merged_test_vs_climo.csv         # if --test
  composite/<mon>/
      climo.csv
      composite_<mon>_mean_std_local_<START>_<END>_<N>stns.png
```

`<IDENT>` is the first four characters of the station filename (e.g., `KDLF`).

---

## `stats.py`

Analyzes station data against a boundary profile to quantify exceedance risks and build plots/PDF reports. Includes optional QC steps to drop flat or unusable days before analysis.

### Data format

**Weather CSV**

- `Date` (MM‑DD‑YYYY)
- `Time (UTC)` (HH:MM)
- `Air Temp (F)`

**Boundary CSV**

- `hour` (0, 0.5, …, 23.5)
- `temp` (°F)

### Usage

```bash
python stats.py \
  --weather ./data/KEDW_ICAO_20150101_20250101.csv \
  --boundary ./data/111FtestCorrected.csv \
  --station KEDW \
  --month 7 \
  --years 2015-2025 \
  --tz America/Los_Angeles \
  --risk2-hours 2 \
  --risk2-area-thresh 10 \
  --outdir ./outputs
```

### Outputs

Saved to `--outdir` with a stem like `KEDW_2015-2024-07_*`:

- `*_risk1_daily_peaks.csv`
- `*_risk2_2h.csv` (or `*_risk2_4h.csv`, etc.) — includes `degree_hours_above_boundary` and `exceed_area_threshold`
- `*_risk1_examples.png`
- `*_risk1_stacked.png` (with boundary **peak** reference line)
- `*_risk2_examples.png` (with per‑day **shifted** boundary in panels)
- `*_risk2_area_examples.png` (area‑based example & non‑example with positive area shaded)
- `*_risk2_stacked.png` (observed curves for days meeting the **window** criterion + one unshifted boundary curve)
- `*_severity_hist.png` (dashed vertical line at area threshold, if set)
- `*_qc_dropped_days.csv` (dates & reasons removed by QC)
- `*_CaseStudy.pdf` (clear % probabilities + 95% CI, plots, and QC summary)

### CLI options (quick reference)

| Option | Meaning |
|---|---|
| `--weather` | Path to weather CSV |
| `--boundary` | Path to boundary CSV (`hour,temp`) |
| `--station` | Label in titles/filenames |
| `--month` | Month 1–12 (repeatable) |
| `--years` | Year(s) or range (`2007`, `2015-2025`, `2015,2017,2020`) |
| `--tz` | IANA zone (e.g., `America/Los_Angeles`) |
| `--risk2-hours` | Window length for Risk 2 (default **2**) |
| `--risk2-area-thresh` | Degree‑hours threshold for area‑based Risk 2 (default **10 °F·h**) |
| `--qc-min-range-f` | Drop days with diurnal range < this (default **2 °F**) |
| `--qc-min-unique` | Drop days with < N unique readings (default **8**) |
| `--qc-max-flat-frac` | Drop days with flat successive readings > frac (default **0.80**) |
| `--qc-min-samples` | Drop days with < N raw samples (default **24**) |
| `--outdir` | Output directory |
| `--title` | Custom PDF title (optional) |

### Interpretation

- **Risk 1:** “Estimated probability is **X %**. With 95 % confidence, true probability lies between **Y %–Z %**.”
- **Risk 2 (window):** same structure; N‑hour window after **peak alignment**.
- **Risk 2 (area):** probability that total positive degree‑hours exceed the chosen threshold (default **10 °F·h**).

### Notes

- Time zone conversion uses `zoneinfo` (`tzdata` may be required).
- Matplotlib runs headless (`Agg` backend).
- PDF generation requires `reportlab`.
- Wilson CI uses the number of days actually evaluated (`n_eval`).

### Troubleshooting

- **Month filter sanity check:**
  ```
  [DEBUG] Filtered records: <rows> | Days: <unique days> | Years: [...] | Month: <m>
  ```
- **Resample mismatch warning:**
  ```
  [WARN] Unique dates from resample/group (X) != calendar day count after QC (Y). Using n_eval=X.
  ```
  Informational; the script uses `n_eval` for all probabilities.
- **PDF image missing:** plots are added only if generated; otherwise the PDF shows “No exceedance events found …”.
- **QC removed everything:** relax thresholds via `--qc-*` or inspect `*_qc_dropped_days.csv`.

## `gui_app.py`

A lightweight Tkinter GUI for the case study generator in `stats.py`.

Launch it with:

```bash
python gui_app.py
```

The **Stats** tab mirrors the CLI options:

- Pick weather and boundary files.
- Enter station, select months and years, choose a timezone.
- Adjust QC thresholds, risk parameters, and output directory.

Click **Run** to generate the analysis; a log area reports success or errors and lists the paths produced by `generate_case_study`.
