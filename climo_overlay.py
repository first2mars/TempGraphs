import argparse
import calendar
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from zoneinfo import ZoneInfo

def build_monthly_climatology(
    raw_csv: str,
    month: int = 8,
    years: int = 10,
    tz_name: str = "America/Chicago",
) -> pd.DataFrame:
    """
    Build hourly climatology (local time) for a given month from climate.af.mil CSV.
    Expected columns in raw_csv: 'Date' (MM-DD-YYYY), 'Time (UTC)' (HH:MM), 'Air Temp (F)'.
    Returns a DataFrame with hour_local, mean, std, min, max, p05, p25, p75, p95.
    """
    df = pd.read_csv(
        raw_csv,
        usecols=["Date", "Time (UTC)", "Air Temp (F)"],
        dtype={"Date": "string", "Time (UTC)": "string"},
        na_values=["", "NA", "NaN", "M", "-", "--", "9999", "9999.9"],
        keep_default_na=True,
        low_memory=False,
    )
    # Combine to UTC datetime
    dt = pd.to_datetime(df["Date"] + " " + df["Time (UTC)"], format="%m-%d-%Y %H:%M", errors="coerce")
    df = df.assign(DATETIME=dt)

    # Filter to requested month
    if not 1 <= int(month) <= 12:
        raise ValueError("month must be an integer from 1..12")
    month = int(month)
    df = df[df["DATETIME"].dt.month == month].copy()
    if df.empty:
        raise ValueError(f"No rows found for month={month} in the provided file.")

    # Filter to most recent years
    latest_year = df["DATETIME"].dt.year.max()
    start_year = latest_year - years + 1
    df = df[df["DATETIME"].dt.year >= start_year].copy()
    if df.empty:
        raise ValueError(f"No rows found for the last {years} years starting from {start_year}.")

    # Convert to tz-aware datetime in UTC
    df["DATETIME"] = df["DATETIME"].dt.tz_localize("UTC")
    # Convert to local tz
    df["DATETIME_LOCAL"] = df["DATETIME"].dt.tz_convert(tz_name)
    # Use INTEGER local hour bins (0..23) for robust aggregation across years
    df["hour_local"] = df["DATETIME_LOCAL"].dt.hour

    # Force temperature to numeric early and drop obviously invalid sentinels
    df["Air Temp (F)"] = pd.to_numeric(df["Air Temp (F)"], errors="coerce")

    # --- Extreme days counts (averaged per year) ---
    # Compute local calendar date
    df["date_local"] = df["DATETIME_LOCAL"].dt.date
    df["year_local"] = df["DATETIME_LOCAL"].dt.year

    # Daily extrema per local date
    daily = df.groupby(["year_local", "date_local"])['Air Temp (F)'].agg(daily_max='max', daily_min='min').reset_index()

    # Count per-year extremes
    def per_year_counts(g):
        dm = g['daily_max']
        dn = g['daily_min']
        return pd.Series({
            'hot_100_109': ((dm >= 100) & (dm <= 109)).sum(),
            'hot_110p': (dm >= 110).sum(),
            'cold_m5_m9': ((dn <= -5) & (dn >= -9)).sum(),
            'cold_m10p': (dn <= -10).sum(),
        })

    yearly = daily.groupby('year_local').apply(per_year_counts, include_groups=False).reset_index()
    # Average number of extreme days per year across the selected period
    extreme_days_avg = yearly.drop(columns=['year_local']).mean(numeric_only=True).to_dict()

    # Group by local hour
    grouped = df.groupby("hour_local")["Air Temp (F)"]

    stats = grouped.agg([
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("p05", lambda x: np.nanpercentile(x, 5) if len(x.dropna()) else np.nan),
        ("p25", lambda x: np.nanpercentile(x, 25) if len(x.dropna()) else np.nan),
        ("p75", lambda x: np.nanpercentile(x, 75) if len(x.dropna()) else np.nan),
        ("p95", lambda x: np.nanpercentile(x, 95) if len(x.dropna()) else np.nan),
    ]).reset_index()

    stats.attrs["start_year"] = start_year
    stats.attrs["latest_year"] = latest_year
    stats.attrs["month"] = month
    stats.attrs["tz_name"] = tz_name
    stats.attrs["extreme_days"] = extreme_days_avg

    return stats

def overlay_and_metrics(test: pd.DataFrame, climo_interp: pd.DataFrame, outdir: str,
                        title_prefix: str = "Del Rio AFB (KDLF)", average_only: bool = False,
                        extreme_days: dict | None = None):
    # Merge test and climo_interp on hour
    merged = pd.merge(test, climo_interp, on="hour", suffixes=("_test", "_climo"))

    # Calculate residuals
    merged["residual"] = merged["temp"] - merged["mean"]

    # Save merged CSV
    merged_csv_path = os.path.join(outdir, "merged_test_vs_climo.csv")
    merged.to_csv(merged_csv_path, index=False)

    # Plot overlay
    x = merged["hour"]
    y_mean = merged["mean"]
    y_std = merged["std"]
    y_test = merged["temp"]
    p05 = merged["p05"]
    p25 = merged["p25"]
    p75 = merged["p75"]
    p95 = merged["p95"]

    plt.figure(figsize=(13, 6.5))
    if not average_only:
        plt.fill_between(x, p05, p95, alpha=0.12, label="5–95%")
        plt.fill_between(x, p25, p75, alpha=0.25, label="25–75% (IQR)")
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.20, label="±1 Std Dev")
    plt.plot(x, y_mean, label="Mean Temp (°F)")
    plt.plot(x, y_test, linewidth=2.5, marker="o", label="Chamber Profile")

    # Optional callout for extreme days (average per year). Only show non-zero categories.
    callout_lines = []
    if extreme_days:
        labels = [
            ("100–109°F", extreme_days.get('hot_100_109', 0)),
            ("≥110°F", extreme_days.get('hot_110p', 0)),
            ("−5 to −9°F", extreme_days.get('cold_m5_m9', 0)),
            ("≤−10°F", extreme_days.get('cold_m10p', 0)),
        ]
        for lab, val in labels:
            if pd.notna(val) and float(val) > 0:
                callout_lines.append(f"{lab}: {float(val):.1f} days/yr")
    if callout_lines:
        txt = "\n".join(callout_lines)
        ax = plt.gca()
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel("Local Hour")
    plt.ylabel("Temperature (°F)")
    plt.legend()
    plt.title(f"Chamber Profile vs {title_prefix}")
    plt.grid(True)
    plt.tight_layout()

    overlay_png_path = os.path.join(outdir, "overlay_chamber_vs_climo.png")
    plt.savefig(overlay_png_path)
    plt.close()

    # Plot residuals
    plt.figure(figsize=(13, 4))
    plt.plot(merged["hour"], merged["residual"], marker="o", linestyle="-")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Local Hour")
    plt.ylabel("Residual (Test - Mean Climo)")
    plt.title(f"Residuals: Chamber Profile - Climatology — {title_prefix}")
    plt.grid(True)
    plt.tight_layout()

    residuals_png_path = os.path.join(outdir, "residuals.png")
    plt.savefig(residuals_png_path)
    plt.close()

def infer_identifier_from_path(path: str) -> str:
    basename = os.path.basename(path)
    ident = basename.split("_")[0]
    return ident

def plot_composite_mean_std(df: pd.DataFrame, out_png: str, title_suffix: str = "", average_only: bool = False,
                            extremes_by_ident: dict[str, dict] | None = None):
    """
    Plot a single mean line per station. If average_only is False, also plot that station's ±1 SD band.
    Expects columns like: hour_local, KDLF_mean, KDLF_std, KEND_mean, KEND_std, ...
    """
    plt.figure(figsize=(14, 7))

    # Identify station identifiers from *_mean columns
    station_ids = sorted({c.split("_")[0] for c in df.columns if c.endswith("_mean")})

    x = df["hour_local"].to_numpy(float)
    for ident in station_ids:
        mean_col = f"{ident}_mean"
        std_col = f"{ident}_std"
        if mean_col not in df.columns:
            continue
        y = df[mean_col].to_numpy(float)
        plt.plot(x, y, label=ident)
        if not average_only and std_col in df.columns:
            s = df[std_col].to_numpy(float)
            plt.fill_between(x, y - s, y + s, alpha=0.12)

    # Build a multi-station callout of extreme-day averages (days/yr) if provided
    if extremes_by_ident:
        lines = []
        for ident in station_ids:
            ex = extremes_by_ident.get(ident, {})
            parts = []
            if ex:
                if ex.get('hot_100_109', 0):
                    parts.append(f"100–109: {ex['hot_100_109']:.1f}")
                if ex.get('hot_110p', 0):
                    parts.append(f"≥110: {ex['hot_110p']:.1f}")
                if ex.get('cold_m5_m9', 0):
                    parts.append(f"−5–−9: {ex['cold_m5_m9']:.1f}")
                if ex.get('cold_m10p', 0):
                    parts.append(f"≤−10: {ex['cold_m10p']:.1f}")
            if parts:
                lines.append(f"{ident}: " + ", ".join(parts))
        if lines:
            txt = "\n".join(lines)
            ax = plt.gca()
            ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontsize=9)

    plt.xlabel("Local Hour")
    plt.ylabel("Temperature (°F)")
    plt.title(f"Composite Climatology {title_suffix}")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Build hourly temperature climatologies and overlay chamber test profiles.")
    ap.add_argument("--raw", help="Path to a station CSV (used to infer defaults and TZ)")
    ap.add_argument("--test", help="Path to chamber test CSV (hour,temp). Required unless --composite.")
    ap.add_argument("--outdir", default="./outputs", help="Output directory root (default: ./outputs)")
    ap.add_argument("--years", type=int, default=10, help="Most recent years to average (default: 10)")
    ap.add_argument("--month", type=int, default=8,
                    help="Month number to build climatology for (1=Jan .. 12=Dec). Default: 8 (August)")
    ap.add_argument("--tz", default="America/Chicago", help="IANA timezone for local hour (default: America/Chicago)")

    ap.add_argument("--composite", action="store_true",
                    help="Composite mode: scan --data_dir for CSVs whose filenames start with a four-letter ID (e.g., KDLF_*.csv) and plot them")
    ap.add_argument("--data_dir", help="Directory to scan for station CSVs")
    ap.add_argument("--composite_test", help="Chamber test CSV to overlay on composite")
    ap.add_argument("--average_only", action="store_true", help="Draw mean lines only (hide ±1 SD bands)")
    ap.add_argument("--stations", help="Comma-separated station identifiers (e.g., KDLF,KEND)")

    args = ap.parse_args()

    if args.composite:
        if not args.data_dir:
            raise ValueError("--data_dir is required for composite mode")

        composite_dir = os.path.join(args.outdir, "composite")
        os.makedirs(composite_dir, exist_ok=True)

        climo_dfs = []
        extremes_by_ident: dict[str, dict] = {}
        # Only accept files that start with a four-letter ID like KDLF_*.csv or KDLF-*.csv
        four_letter_pattern = re.compile(r"^[A-Za-z]{4}[_-].*\.csv$")
        for raw_name in sorted(os.listdir(args.data_dir)):
            # Must be CSV and start with a four-letter identifier prefix
            if not raw_name.endswith(".csv"):
                continue
            if not four_letter_pattern.match(raw_name):
                print(f"[INFO] Skipping non-station-named CSV (expects 4-letter prefix): {os.path.join(args.data_dir, raw_name)}")
                continue

            # Optional explicit station filter
            if args.stations:
                idents = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
                if raw_name[:4].upper() not in idents:
                    print(f"[INFO] Excluding station {raw_name[:4].upper()} (not in --stations)")
                    continue

            raw_path = os.path.join(args.data_dir, raw_name)
            try:
                cl = build_monthly_climatology(raw_path, month=args.month, years=args.years, tz_name=args.tz)
            except Exception as e:
                print(f"[WARN] Skipping {raw_path}: {e}")
                continue

            ident_i = infer_identifier_from_path(raw_path)
            # Keep station-specific extreme-day averages
            extremes_by_ident[ident_i] = cl.attrs.get('extreme_days', {})

            mon_abbr = calendar.month_abbr[int(args.month)].lower()
            station_outdir_i = os.path.join(args.outdir, ident_i)
            os.makedirs(station_outdir_i, exist_ok=True)
            per_csv_path = os.path.join(
                station_outdir_i,
                f"{mon_abbr}_climatology_local_{cl.attrs.get('start_year','?')}_{cl.attrs.get('latest_year','?')}.csv",
            )
            cl.to_csv(per_csv_path, index=False)
            climo_dfs.append(cl.rename(columns=lambda c: f"{ident_i}_{c}" if c != "hour_local" else c))

        if not climo_dfs:
            print("No station data processed for composite.")
            return

        composite_df = reduce(lambda left, right: pd.merge(left, right, on="hour_local", how="outer"), climo_dfs)
        composite_df = composite_df.sort_values("hour_local").reset_index(drop=True)

        start_year = min(cl.attrs.get("start_year", 0) for cl in climo_dfs)
        latest_year = max(cl.attrs.get("latest_year", 0) for cl in climo_dfs)

        mon_abbr = calendar.month_abbr[int(args.month)].lower()
        composite_csv = os.path.join(
            composite_dir,
            f"composite_{mon_abbr}_climo_local_{start_year}_{latest_year}.csv",
        )
        composite_df.to_csv(composite_csv, index=False)

        composite_png = os.path.join(
            composite_dir,
            f"composite_{mon_abbr}_mean_std_local_{start_year}_{latest_year}.png",
        )
        title_suffix = f"{calendar.month_name[int(args.month)]} ({start_year}–{latest_year})"

        plot_composite_mean_std(
            composite_df, composite_png,
            title_suffix=title_suffix,
            average_only=args.average_only,
            extremes_by_ident=extremes_by_ident,
        )

        if args.composite_test:
            test = pd.read_csv(
                args.composite_test,
                dtype={"hour": "float64", "temp": "float64"},
                low_memory=False,
            )
            test.columns = [c.strip().lower() for c in test.columns]
            # Additional composite test overlay code would go here (not specified)

    else:
        if not args.raw:
            raise ValueError("--raw is required for single-station mode")

        climo = build_monthly_climatology(args.raw, month=args.month, years=args.years, tz_name=args.tz)

        ident = infer_identifier_from_path(args.raw)
        station_outdir = os.path.join(args.outdir, ident)
        os.makedirs(station_outdir, exist_ok=True)

        mon_abbr = calendar.month_abbr[int(args.month)].lower()
        climo_csv = os.path.join(
            station_outdir,
            f"{mon_abbr}_climatology_local_{climo.attrs.get('start_year','?')}_{climo.attrs.get('latest_year','?')}.csv",
        )
        climo.to_csv(climo_csv, index=False)

        if args.test:
            test = pd.read_csv(
                args.test,
                dtype={"hour": "float64", "temp": "float64"},
                low_memory=False,
            )
            # Normalize headers just in case
            test.columns = [c.strip().lower() for c in test.columns]
            # Ensure required columns are present and numeric
            if not {"hour", "temp"}.issubset(set(test.columns)):
                raise ValueError("Test CSV must have columns: hour,temp")
            test = test.dropna(subset=["hour", "temp"]).sort_values("hour").reset_index(drop=True)

            # Ensure climo numeric for interpolation
            for col in ["hour_local", "mean", "std", "min", "max", "p05", "p25", "p75", "p95"]:
                climo[col] = pd.to_numeric(climo[col], errors="coerce")

            # Interpolate climo to test hours
            climo_interp = pd.DataFrame()
            climo_interp["hour"] = test["hour"]
            climo_interp["mean"] = np.interp(test["hour"], climo["hour_local"], climo["mean"])
            climo_interp["std"] = np.interp(test["hour"], climo["hour_local"], climo["std"])
            climo_interp["min"] = np.interp(test["hour"], climo["hour_local"], climo["min"])
            climo_interp["max"] = np.interp(test["hour"], climo["hour_local"], climo["max"])
            climo_interp["p05"] = np.interp(test["hour"], climo["hour_local"], climo["p05"])
            climo_interp["p25"] = np.interp(test["hour"], climo["hour_local"], climo["p25"])
            climo_interp["p75"] = np.interp(test["hour"], climo["hour_local"], climo["p75"])
            climo_interp["p95"] = np.interp(test["hour"], climo["hour_local"], climo["p95"])

            title_prefix = f"{calendar.month_name[int(args.month)]} Climatology • {args.tz} • {climo.attrs.get('start_year','?')}-{climo.attrs.get('latest_year','?')}"
            overlay_and_metrics(
                test, climo_interp, station_outdir,
                title_prefix=title_prefix,
                average_only=args.average_only,
                extreme_days=climo.attrs.get('extreme_days')
            )

if __name__ == "__main__":
    main()