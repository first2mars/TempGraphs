import argparse
import calendar
from typing import Optional
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

def build_monthly_climatology(
    raw_csv: str,
    month: int = 8,
    years: int = 10,
    tz_name: str = "America/Chicago",
    start_year: int | None = None,
    end_year: int | None = None,
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

    # Select year window: explicit start/end if provided, else last N years ending at data max
    year_series = df["DATETIME"].dt.year
    data_min_year = int(year_series.min())
    data_max_year = int(year_series.max())

    if start_year is not None or end_year is not None:
        sy = int(start_year) if start_year is not None else max(data_min_year, data_max_year - years + 1)
        ey = int(end_year) if end_year is not None else data_max_year
        if sy > ey:
            raise ValueError(f"Invalid year range: start_year ({sy}) > end_year ({ey}).")
    else:
        ey = data_max_year
        sy = ey - years + 1

    # clamp to available data
    sy = max(sy, data_min_year)
    ey = min(ey, data_max_year)

    df = df[(year_series >= sy) & (year_series <= ey)].copy()
    if df.empty:
        raise ValueError(f"No rows found within year range {sy}-{ey}.")

    latest_year = ey
    start_year = sy

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
                        extreme_days: dict | None = None,
                        overlay_png_name: str | None = None,
                        residuals_png_name: str | None = None):
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
        plt.fill_between(x, p05, p95, alpha=0.12, label="5–95%", zorder=1)
        plt.fill_between(x, p25, p75, alpha=0.25, label="25–75% (IQR)", zorder=1)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.20, label="±1 Std Dev", zorder=1)
    plt.plot(x, y_mean, label="Mean Temp (°F)", zorder=2)
    plt.plot(x, y_test, linewidth=2.5, marker="o", label="Chamber Profile", zorder=3)

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
        # Draw a small label ABOVE the box
        ax = plt.gca()
        ax.text(
            0.98, 0.18, "Average days per year",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontweight='bold',
            zorder=10, clip_on=False
        )
        # Then draw the box with ONLY the values
        txt = "\n".join(callout_lines)
        ax.text(
            0.98, 0.06, txt,
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            zorder=10, clip_on=False
        )

    plt.xlabel("Local Hour")
    plt.ylabel("Temperature (°F)")
    plt.legend()
    plt.title(f"Chamber Profile vs {title_prefix}")
    plt.grid(True)
    plt.tight_layout()

    overlay_png_path = os.path.join(outdir, overlay_png_name or "overlay.png")
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
    residuals_png_path = os.path.join(outdir, residuals_png_name or "residuals.png")
    plt.savefig(residuals_png_path)
    plt.close()

def infer_identifier_from_path(path: str) -> str:
    basename = os.path.basename(path)
    m = re.match(r"^([A-Za-z]{4})", basename)
    return m.group(1).upper() if m else basename[:4].upper()



def plot_composite_mean_std(
    df: pd.DataFrame,
    out_png: str,
    title_suffix: str = "",
    average_only: bool = False,
    extremes_by_ident: dict[str, dict] | None = None,
    test: pd.DataFrame | None = None,
):
    """
    Plot a single mean line per station. If average_only is False, also shade 25–75% (IQR) and 5–95% ranges.
    Expects columns like: hour_local, KDLF_mean, KDLF_p25, KDLF_p75, KDLF_p05, KDLF_p95, ...
    """
    plt.figure(figsize=(14, 7))

    # Identify station identifiers from *_mean columns
    station_ids = sorted({c.split("_")[0] for c in df.columns if c.endswith("_mean")})

    x = df["hour_local"].to_numpy(float)
    drew_iqr_band = False
    drew_p95_band = False
    for ident in station_ids:
        mean_col = f"{ident}_mean"
        std_col = f"{ident}_std"
        if mean_col not in df.columns:
            continue
        y = df[mean_col].to_numpy(float)
        plt.plot(x, y, label=ident, zorder=2)
        if not average_only:
            p25_col = f"{ident}_p25"
            p75_col = f"{ident}_p75"
            p05_col = f"{ident}_p05"
            p95_col = f"{ident}_p95"
            # 25–75% (IQR)
            if p25_col in df.columns and p75_col in df.columns:
                q25 = df[p25_col].to_numpy(float)
                q75 = df[p75_col].to_numpy(float)
                plt.fill_between(x, q25, q75, alpha=0.25, zorder=1)
                drew_iqr_band = True
            # 5–95% range
            if p05_col in df.columns and p95_col in df.columns:
                q05 = df[p05_col].to_numpy(float)
                q95 = df[p95_col].to_numpy(float)
                plt.fill_between(x, q05, q95, alpha=0.12, zorder=1)
                drew_p95_band = True

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
            ax = plt.gca()
            # Draw label ABOVE the box
            ax.text(
                0.98, 0.18, "Average days per year",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontweight='bold',
                zorder=10, clip_on=False
            )
            # Then the box with ONLY the values
            txt = "\n".join(lines)
            ax.text(
                0.98, 0.06, txt,
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
                fontsize=9,
                zorder=10, clip_on=False
            )

    # Optional overlay of chamber test profile
    if test is not None:
        cols = {str(c).strip().lower() for c in test.columns}
        if {"hour", "temp"}.issubset(cols):
            x_t = pd.to_numeric(test["hour"], errors="coerce").to_numpy()
            y_t = pd.to_numeric(test["temp"], errors="coerce").to_numpy()
            mask = np.isfinite(x_t) & np.isfinite(y_t)
            if mask.any():
                plt.plot(
                    x_t[mask], y_t[mask],
                    linewidth=2.8, marker="o",
                    label="Chamber Profile", zorder=3
                )

    plt.xlabel("Local Hour")
    plt.ylabel("Temperature (°F)")
    plt.title(f"Composite Climatology {title_suffix}")
    ax = plt.gca()
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    # Add patches for shaded regions if drawn
    if drew_iqr_band:
        handles.append(Patch(facecolor='gray', alpha=0.25, label='25–75% (IQR)'))
    if drew_p95_band:
        handles.append(Patch(facecolor='gray', alpha=0.12, label='5–95% Range'))
    if len(handles) != len(labels):
        # We added at least one patch
        ax.legend(handles, [h.get_label() for h in handles], ncol=2)
    else:
        ax.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# Resolve latest station CSV in a directory by 4-letter identifier prefix
# Chooses the lexicographically last filename that matches, which typically
# corresponds to the most recent export pattern.
def find_latest_station_csv(data_dir: str, ident: str) -> str | None:
    ident = ident.strip().upper()
    pat = re.compile(rf"^{ident}[_-].*\.csv$", re.IGNORECASE)
    try:
        candidates = [n for n in os.listdir(data_dir) if pat.match(n)]
    except FileNotFoundError:
        return None
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(data_dir, candidates[-1])

def compute_per_year_hourly_means(
    raw_csv: str,
    month: int,
    tz_name: str,
    years: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Return a DataFrame [year, hour, mean_temp] for the chosen month, and the
    (start_year, end_year) actually used after clamping to data.
    """
    df = pd.read_csv(raw_csv)
    need = {"Date", "Time (UTC)", "Air Temp (F)"}
    if not need.issubset(df.columns):
        raise ValueError("Input CSV must contain 'Date', 'Time (UTC)', 'Air Temp (F)'.")

    from datetime import timezone
    try:
        from zoneinfo import ZoneInfo  # py3.9+
    except Exception:
        ZoneInfo = None

    def _to_local(row):
        dt_utc = datetime.strptime(f"{row['Date']} {row['Time (UTC)']}", "%m-%d-%Y %H:%M").replace(tzinfo=timezone.utc)
        if ZoneInfo is None:
            return dt_utc
        try:
            return dt_utc.astimezone(ZoneInfo(tz_name))
        except Exception:
            return dt_utc

    df["dt_local"] = df.apply(_to_local, axis=1)
    df = df[df["dt_local"].dt.month == int(month)].copy()
    if df.empty:
        raise ValueError(f"No data for month {month} after timezone conversion.")

    ys = df["dt_local"].dt.year
    data_min, data_max = int(ys.min()), int(ys.max())

    if start_year is None and end_year is None:
        ey = data_max
        sy = ey - int(years) + 1
    else:
        sy = int(start_year) if start_year is not None else max(data_min, data_max - int(years) + 1)
        ey = int(end_year) if end_year is not None else data_max

    sy = max(sy, data_min)
    ey = min(ey, data_max)

    df = df[(ys >= sy) & (ys <= ey)].copy()
    if df.empty:
        raise ValueError(f"No rows within requested year window {sy}-{ey}.")

    df["hour_local"] = df["dt_local"].dt.hour.astype(int)
    df["year"] = df["dt_local"].dt.year.astype(int)

    g = (
        df.groupby(["year", "hour_local"], as_index=False)["Air Temp (F)"]
        .mean()
        .rename(columns={"hour_local": "hour", "Air Temp (F)": "mean_temp"})
    )

    # ensure 0..23 per year, interpolate if an hour is missing
    filled = []
    for yr, sub in g.groupby("year"):
        sub = sub.set_index("hour").reindex(range(24))
        sub.index.name = "hour"
        sub["year"] = int(yr)
        sub["mean_temp"] = sub["mean_temp"].interpolate(limit_direction="both")
        filled.append(sub.reset_index())

    out = pd.concat(filled, ignore_index=True)
    return out[["year", "hour", "mean_temp"]], sy, ey


def plot_per_year_means(
    df_year_hour: pd.DataFrame,
    ident: str,
    month: int,
    sy: int,
    ey: int,
    test_csv: Optional[str],
    outdir: str,
):
    """Plot one mean line per year; optional chamber overlay."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for yr, sub in df_year_hour.groupby("year"):
        sub = sub.sort_values("hour")
        ax.plot(sub["hour"].values, sub["mean_temp"].values, linewidth=1.8, label=str(int(yr)), zorder=2)

    if test_csv:
        try:
            test = pd.read_csv(test_csv)
            h = test.iloc[:, 0].astype(float).values
            t = test.iloc[:, 1].astype(float).values
            ax.plot(h, t, linewidth=2.5, linestyle="--", label="Chamber Profile", zorder=3)
        except Exception as e:
            print(f"[WARN] Could not overlay test profile: {e}")

    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour (local)")
    ax.set_ylabel("Temperature (°F)")
    ax.set_title(f"{ident} — {calendar.month_name[int(month)]} Per-Year Hourly Means • {sy}–{ey}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)

    fname = f"{ident}_{calendar.month_abbr[int(month)].lower()}_peryear_means_{sy}_{ey}.png"
    out_path = os.path.join(outdir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Build hourly temperature climatologies and overlay chamber test profiles.")
    ap.add_argument("--raw", help="Path to a station CSV (used to infer defaults and TZ)")
    ap.add_argument("--test", help="Path to chamber test CSV (hour,temp). Required unless --composite.")
    ap.add_argument("--outdir", default="./outputs", help="Output directory root (default: ./outputs)")
    ap.add_argument("--years", type=int, default=10, help="Most recent years to average (default: 10)")
    ap.add_argument("--start_year", type=int, help="Start year (inclusive) for the climatology window. Overrides --years when set.")
    ap.add_argument("--end_year", type=int, help="End year (inclusive) for the climatology window. Overrides --years when set.")
    ap.add_argument("--month", type=int, default=8,
                    help="Month number to build climatology for (1=Jan .. 12=Dec). Default: 8 (August)")
    ap.add_argument("--tz", default="America/Chicago", help="IANA timezone for local hour (default: America/Chicago)")

    ap.add_argument("--composite", action="store_true",
                    help="Composite mode: scan --data_dir for CSVs whose filenames start with a four-letter ID (e.g., KDLF_*.csv) and plot them")
    ap.add_argument("--data_dir", help="Directory to scan for station CSVs")
    ap.add_argument("--composite_test", help="Chamber test CSV to overlay on composite")
    ap.add_argument("--average_only", action="store_true", help="Draw mean lines only (hide ±1 SD bands)")
    ap.add_argument("--stations", help="Comma-separated station identifiers (e.g., KDLF,KEND)")
    ap.add_argument("--station", help="Single-station mode: comma-separated 4-letter IDs (e.g., KDLF or KDLF,KEND). Requires --data_dir if --raw is not provided. If both --raw and --station are provided, --raw takes priority.")
    ap.add_argument(
    "--per_year",
    action="store_true",
    help="Single-station: plot a separate mean line for each year in the selected window (no shaded bands)."
    )
    args = ap.parse_args()

    if args.composite:
        if not args.data_dir:
            raise ValueError("--data_dir is required for composite mode")

        mon_abbr = calendar.month_abbr[int(args.month)].lower()
        composite_dir = os.path.join(args.outdir, "composite", mon_abbr)
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
                cl = build_monthly_climatology(
                    raw_path,
                    month=args.month,
                    years=args.years,
                    tz_name=args.tz,
                    start_year=args.start_year,
                    end_year=args.end_year,
                )
            except Exception as e:
                print(f"[WARN] Skipping {raw_path}: {e}")
                continue

            ident_i = infer_identifier_from_path(raw_path)
            # Keep station-specific extreme-day averages
            extremes_by_ident[ident_i] = cl.attrs.get('extreme_days', {})

            station_outdir_i = os.path.join(args.outdir, "stations", ident_i, mon_abbr)
            os.makedirs(station_outdir_i, exist_ok=True)
            per_csv_path = os.path.join(station_outdir_i, "climo.csv")
            cl.to_csv(per_csv_path, index=False)
            climo_dfs.append(cl.rename(columns=lambda c: f"{ident_i}_{c}" if c != "hour_local" else c))

        if not climo_dfs:
            print("No station data processed for composite.")
            return

        composite_df = reduce(lambda left, right: pd.merge(left, right, on="hour_local", how="outer"), climo_dfs)
        composite_df = composite_df.sort_values("hour_local").reset_index(drop=True)

        start_year = min(cl.attrs.get("start_year", 0) for cl in climo_dfs)
        latest_year = max(cl.attrs.get("latest_year", 0) for cl in climo_dfs)

        composite_csv = os.path.join(composite_dir, "climo.csv")
        composite_df.to_csv(composite_csv, index=False)

        nstations = len(extremes_by_ident) if extremes_by_ident else len([c for c in composite_df.columns if c.endswith("_mean")])
        composite_png = os.path.join(
            composite_dir,
            f"composite_{mon_abbr}_mean_std_local_{start_year}_{latest_year}_{nstations}stns.png",
        )
        title_suffix = f"{calendar.month_name[int(args.month)]} ({start_year}–{latest_year})"

        test_df_for_composite = None
        if args.composite_test:
            test_df_for_composite = pd.read_csv(
                args.composite_test,
                dtype={"hour": "float64", "temp": "float64"},
                low_memory=False,
            )
            test_df_for_composite.columns = [c.strip().lower() for c in test_df_for_composite.columns]
            print(f"[INFO] Composite test loaded: {len(test_df_for_composite)} rows from {args.composite_test}")

        plot_composite_mean_std(
            composite_df, composite_png,
            title_suffix=title_suffix,
            average_only=args.average_only,
            extremes_by_ident=extremes_by_ident,
            test=test_df_for_composite,
        )

    else:
        # Determine which station files to run in single-station mode
        raw_paths: list[str] = []
        if args.raw:
            raw_paths = [args.raw]
        elif args.station:
            if not args.data_dir:
                raise ValueError("--data_dir is required when using --station in single-station mode")
            idents = [tok.strip().upper() for tok in args.station.split(',') if tok.strip()]
            for ident_req in idents:
                resolved = find_latest_station_csv(args.data_dir, ident_req)
                if not resolved:
                    print(f"[WARN] No CSV found in {args.data_dir} for station {ident_req}")
                    continue
                raw_paths.append(resolved)
        else:
            raise ValueError("Provide either --raw or --station for single-station mode")

        if not raw_paths:
            print("No station files to process in single-station mode.")
            return

        # Process each selected station file
        for raw_path in raw_paths:
            climo = build_monthly_climatology(
                raw_path,
                month=args.month,
                years=args.years,
                tz_name=args.tz,
                start_year=args.start_year,
                end_year=args.end_year,
            )

            ident = infer_identifier_from_path(raw_path)
            mon_abbr = calendar.month_abbr[int(args.month)].lower()
            station_outdir = os.path.join(args.outdir, "stations", ident, mon_abbr)
            os.makedirs(station_outdir, exist_ok=True)

            # Save per-station climatology with simple name
            climo_csv = os.path.join(station_outdir, "climo.csv")
            climo.to_csv(climo_csv, index=False)

            # Optional per-year plot (independent of overlay)
            if args.per_year:
                df_yh, sy_used, ey_used = compute_per_year_hourly_means(
                    raw_path,
                    month=args.month,
                    tz_name=args.tz,
                    years=args.years,
                    start_year=args.start_year,
                    end_year=args.end_year,
                )
                per_year_png = plot_per_year_means(
                    df_yh,
                    ident=ident,
                    month=args.month,
                    sy=sy_used,
                    ey=ey_used,
                    test_csv=args.test if args.test else None,
                    outdir=station_outdir,
                )
                print(f"Saved per-year means plot: {per_year_png}")

            # Overlay + residuals only if a test profile is provided
            if args.test:
                test = pd.read_csv(
                    args.test,
                    dtype={"hour": "float64", "temp": "float64"},
                    low_memory=False,
                )
                test.columns = [c.strip().lower() for c in test.columns]
                if not {"hour", "temp"}.issubset(set(test.columns)):
                    raise ValueError("Test CSV must have columns: hour,temp")
                test = test.dropna(subset=["hour", "temp"]).sort_values("hour").reset_index(drop=True)

                for col in ["hour_local", "mean", "std", "min", "max", "p05", "p25", "p75", "p95"]:
                    climo[col] = pd.to_numeric(climo[col], errors="coerce")

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

                title_prefix = f"{ident} — {calendar.month_name[int(args.month)]} Climatology • {climo.attrs.get('start_year','?')}-{climo.attrs.get('latest_year','?')}"
                overlay_png_name = f"{ident}_{mon_abbr}_overlay.png"
                residuals_png_name = f"{ident}_{mon_abbr}_residuals.png"

                overlay_and_metrics(
                    test, climo_interp, station_outdir,
                    title_prefix=title_prefix,
                    average_only=args.average_only,
                    extreme_days=climo.attrs.get('extreme_days'),
                    overlay_png_name=overlay_png_name,
                    residuals_png_name=residuals_png_name,
                )

if __name__ == "__main__":
    main()