Here’s a clean composite example (no --raw, filters to KDLF & KEND, mean lines only, and overlays your chamber profile):

python climo_overlay.py \
  --outdir "./outputs" \
  --data_dir "./data" \
  --years 10 \
  --month 8 \
  --composite \
  --average_only \
  --stations "KDLF,KEND" \
  --composite_test "./data/111Ftest.csv"

And here’s an updated README.md you can drop in that captures everything we’ve built so far — flags, output structure, examples, and gotchas.

⸻

SimpleWeather — Hourly Climatology + Chamber Overlay

A utility to build monthly hourly temperature climatologies from climate.af.mil station CSVs (converted to local time), and to overlay a chamber test profile. Also supports composite plots across many stations.

What it does
	•	Monthly climatology per station (you choose the month):
	•	Per local hour (0–23): mean, std, min, max, p05, p25, p75, p95
	•	Uses the most recent N years found in the file (default 10)
	•	Overlay a chamber test (hour,temp; 24 or 48 points). Climo is interpolated to your test hours.
	•	Composite mode across a folder:
	•	Auto-discovers CSVs whose filenames start with a four-letter ID (e.g., KDLF_*.csv or KDLF-*.csv)
	•	Draws one mean line per station, with optional ±1 SD band
	•	Can overlay your test profile on the composite
	•	Extreme-days callouts (for the selected month), showing the average days/year in these bins (shown only if non-zero):
	•	100–109°F, ≥110°F, −5 to −9°F, ≤−10°F

⸻

Requirements & setup

Python 3.9+ recommended.

pandas>=2.0.0
numpy>=1.25.0
matplotlib>=3.7.0
tzdata>=2023.3
backports.zoneinfo; python_version < "3.9"

Create a venv and install:

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt


⸻

Input formats

Station CSV (from climate.af.mil)

Columns required:
	•	Date in MM-DD-YYYY
	•	Time (UTC) in HH:MM
	•	Air Temp (F) numeric

Climatology is computed for your selected month using local time.

Chamber test CSV
	•	Headers not case sensitive; required columns:
	•	hour → 0, 0.5, 1.0, …, 23.5 (local time)
	•	temp → °F

⸻

Usage

Single-station (by path)

python climo_overlay.py \
  --raw "./data/KDLF_ICAO_20150101_20250101.csv" \
  --test "./data/111Ftest.csv" \
  --outdir "./outputs" \
  --years 10 \
  --month 8 \
  # --average_only           # optional: mean line only (no bands)

Single-station (by 4-letter ID, or multiple IDs)

Finds the latest matching CSV(s) in --data_dir:

python climo_overlay.py \
  --station "KDLF,KEND" \
  --data_dir "./data" \
  --test "./data/111Ftest.csv" \
  --outdir "./outputs" \
  --years 10 \
  --month 8 \
  # --average_only

Composite across many stations (auto-discover)

Scans --data_dir for files beginning with a 4-letter ID:

python climo_overlay.py \
  --outdir "./outputs" \
  --data_dir "./data" \
  --years 10 \
  --month 8 \
  --composite \
  # --average_only \
  # --stations "KDLF,KEND" \
  # --composite_test "./data/111Ftest.csv"


⸻

Output structure

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

	•	<IDENT> is the first 4 characters of the station filename (e.g., KDLF).
	•	<mon> is the lowercase month abbreviation (e.g., aug).
	•	<START>_<END> is the climatology window (e.g., 2015_2024).
	•	Composite PNG includes the month, year span, and station count.

⸻

Flags (quick reference)

--raw             Path to a station CSV (single-station mode)
--station         Comma-separated 4-letter IDs (single-station mode; uses --data_dir)
--test            Chamber test CSV (hour,temp). Required unless --composite.
--outdir          Output directory root (default: ./outputs)
--data_dir        Directory containing station CSVs (for --station & composite)
--years           Most recent years to average (default: 10)
--month           Month number (1=Jan .. 12=Dec). Default: 8 (Aug)
--tz              IANA timezone for local hour (default: America/Chicago)

--composite       Enable composite mode (scan --data_dir)
--stations        Comma-separated station IDs to include (composite filter)
--composite_test  Chamber test CSV to overlay on composite
--average_only    Mean line only (hide ±1 SD bands) — applies to both modes


⸻

What the charts show
	•	Single-station overlay
	•	Mean line (always)
	•	Bands (unless --average_only): ±1 SD, IQR (25–75%), 5–95%
	•	Chamber Profile line (from --test)
	•	Callout of average extreme days for the station/month (only non-zero bins)
	•	Residuals bar chart: (test − climatology mean) by hour
	•	Composite
	•	One mean line per station
	•	Optional ±1 SD band per station (omit with --average_only)
	•	Callout summarizing each station’s non-zero extreme-day bins
	•	Optional Chamber Profile overlay (--composite_test)

⸻

Notes & robustness
	•	Stats aggregate by integer local hour (0–23) to avoid over-fragmenting bins.
	•	Percentiles use NaN-safe nanpercentile guards.
	•	CSV reads use explicit dtypes and low_memory=False to avoid mixed-type pitfalls.
	•	Composite mode skips non-station files and logs an info message.

⸻

Troubleshooting
	•	KeyError: 'hour'
Ensure your test CSV has hour,temp and hour is numeric (e.g., 0, 0.5, …, 23.5).
	•	Dtype warnings
We set explicit dtypes; if you still see warnings, double-check your station CSV columns and header row.
	•	Empty month
Make sure the station CSV contains data for the --month you selected.
	•	Composite skipped file
Filenames must begin with a 4-letter ID (e.g., KDLF_*). Test profiles like 111Ftest.csv are skipped on purpose.

⸻

Changelog (recent)
	•	Added --month (any month)
	•	--average_only now works in both single-station and composite
	•	Composite no longer needs --raw; auto-discovers by 4-letter prefix & supports --stations
	•	Single-station supports --station (ID or comma-separated list) via --data_dir
	•	Added extreme-days callouts (100–109, ≥110, −5 to −9, ≤−10)
	•	New output layout and descriptive PNG names
	•	Stricter dtype handling & NaN-safe percentiles

⸻

If you want, I can also add a --legend_sort option (e.g., by max temp or by 3 pm temp) for composites with lots of stations, or a --month Aug parser that accepts names as well as numbers.
