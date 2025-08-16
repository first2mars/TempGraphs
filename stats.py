from __future__ import annotations

try:  # Optional PDF dependencies
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet  # type: ignore

    styles = getSampleStyleSheet()
    _HAVE_REPORTLAB = True
except Exception:  # pragma: no cover - environment without reportlab
    Image = Paragraph = SimpleDocTemplate = Spacer = letter = None  # type: ignore
    styles = None
    _HAVE_REPORTLAB = False

if _HAVE_REPORTLAB:
    # Helper: Add plot image to PDF story if available, else fallback text.
    def add_plot_or_text(story, plot_path, description, width=400, height=250):
        """Add an image to the story or fallback text if missing."""
        if plot_path:
            story.append(Image(plot_path, width=width, height=height))
        else:
            story.append(
                Paragraph(
                    f"<b>No exceedance events found for {description}.</b>",
                    styles["Normal"],
                )
            )
        story.append(Spacer(1, 12))
else:
    def add_plot_or_text(story, plot_path, description, width=400, height=250):
        """Fallback when reportlab is unavailable; append plain text."""
        if plot_path:
            story.append(f"[Image: {plot_path}]")
        else:
            story.append(f"No exceedance events found for {description}.")

from pathlib import Path
def plot_risk2_area_examples(df, shifted_boundary, station, month, outdir, theta_area, direction="above"):
    """
    Plot example days showing exceedance and non-exceedance for area-under-curve thermal load.
    direction: "above" (hot) or "below" (cold); controls whether area is computed above or below the boundary.
    """
    local_hours = np.arange(24)
    exceed_days = []
    non_exceed_days = []

    # Compute area exceedance for each day via np.trapezoid
    for date, group in df.groupby(df.index.date):
        observed = group["temp"].values
        if direction == "above":
            area = np.trapezoid(np.clip(observed - shifted_boundary, 0, None), local_hours)
        else:
            area = np.trapezoid(np.clip(shifted_boundary - observed, 0, None), local_hours)
        if area > theta_area and len(exceed_days) < 1:
            exceed_days.append((date, observed, area))
        elif area <= theta_area and len(non_exceed_days) < 1:
            non_exceed_days.append((date, observed, area))
        if len(exceed_days) and len(non_exceed_days):
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if exceed_days:
        date, observed, area = exceed_days[0]
        axes[0].plot(local_hours, observed, label=f"Observed {date}", color="blue")
        axes[0].plot(local_hours, shifted_boundary, label="Shifted Boundary", color="red")
        cond0 = (observed > shifted_boundary) if direction == "above" else (observed < shifted_boundary)
        axes[0].fill_between(local_hours, shifted_boundary, observed, where=cond0, color="blue", alpha=0.3)
        axes[0].set_title("Exceedance Example")
        axes[0].legend()
        axes[0].annotate(f"A⁺ = {area:.1f} °F·h\nθ = {theta_area} °F·h",
                         xy=(0.05, 0.9), xycoords="axes fraction", fontsize=10,
                         bbox=dict(facecolor="white", alpha=0.7))

    if non_exceed_days:
        date, observed, area = non_exceed_days[0]
        axes[1].plot(local_hours, observed, label=f"Observed {date}", color="blue")
        axes[1].plot(local_hours, shifted_boundary, label="Shifted Boundary", color="red")
        cond1 = (observed > shifted_boundary) if direction == "above" else (observed < shifted_boundary)
        axes[1].fill_between(local_hours, shifted_boundary, observed, where=cond1, color="blue", alpha=0.3)
        axes[1].set_title("Non-exceedance Example")
        axes[1].legend()
        axes[1].annotate(f"A⁺ = {area:.1f} °F·h\nθ = {theta_area} °F·h",
                         xy=(0.05, 0.9), xycoords="axes fraction", fontsize=10,
                         bbox=dict(facecolor="white", alpha=0.7))

    for ax in axes:
        ax.set_xlabel("Local Hour")
        ax.set_ylabel("Temp (°F)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    outfile = Path(outdir) / f"{station}_{month:02d}_risk2_area_examples.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile
"""
stats.py — Temperature vs Boundary Case Study Generator

Purpose
-------
Given:
  • a weather CSV with columns like [Date, Time (UTC), Air Temp (F)]
  • a boundary CSV with columns: hour (0..23.5 by 0.5), temp (°F)
this module produces:
  • Risk 1 and Risk 2 probabilities (with 95% Wilson CIs)
  • CSVs with day‑by‑day results
  • Plots (examples, stacked curves, severity histogram)
  • A polished PDF report with clear probability statements

Key definitions
---------------
Risk 1: Daily max observed temperature > boundary peak temperature (no time shift).
Risk 2: After peak‑aligning boundary to observed day, any contiguous N‑hour window
        with observed > boundary (default N = 2 h). Also reports degree‑hours.

This version extends the original functionality to allow aggregation across
multiple years. Users can specify a single year, a comma‑separated list of years,
or inclusive ranges such as ``2015-2025``. The script will combine all days
for the selected month and years into a single analysis.

Usage (CLI)
-----------
python stats.py \
  --weather /path/KEDW_ICAO_20150101_20250101.csv \
  --boundary /path/111FtestCorrected.csv \
  --station KEDW \
  --month 7 \
  --years 2015-2025 \
  --tz America/Los_Angeles \
  --risk2-hours 2 \
  --outdir ./outputs

You can repeat ``--month`` to run multiple case studies in one call, e.g.:
  --month 1 --month 7 --years 2015-2025

Dependencies
------------
  pandas, numpy, matplotlib, reportlab (for PDF)
"""

import argparse
import calendar
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import matplotlib

# Use non‑interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None


@dataclass
class CaseStudyOutputs:
    pdf: str
    csv_risk1: str
    csv_risk2: str
    plot_risk1_examples: str | None
    plot_risk1_stacked: str
    plot_risk2_examples: str | None
    plot_risk2_stacked: str | None
    plot_severity_hist: str
    stats: Dict


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[Tuple[float, float], float]:
    """Wilson score interval (95% by default) and point estimate."""
    if n == 0:
        return (float("nan"), float("nan")), float("nan")
    if k < 0:
        k = 0
    if k > n:
        # clamp and warn in console; prevents sqrt domain errors if upstream mismatch occurs
        print(f"[WARN] clamping k ({k}) to n ({n}) in wilson_ci")
        k = n
    phat = k / n
    den = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / den
    half = z * math.sqrt((phat * (1 - phat) / n) + (z ** 2 / (4 * n ** 2))) / den
    return (center - half, center + half), phat


def parse_years_input(year_tokens: List[str] | None) -> List[int]:
    """
    Parse a list of year tokens into a sorted list of unique years.
    Each token may be a single year (e.g. "2007"), a comma‑separated list
    ("2007,2010"), or an inclusive range ("2015-2020").
    """
    years: List[int] = []
    if not year_tokens:
        return years
    for token in year_tokens:
        if not token:
            continue
        # Split on comma to allow multiple entries in one argument
        for sub in str(token).split(','):
            sub = sub.strip()
            if not sub:
                continue
            # Extract all integers from this substring using regex. This makes
            # parsing more robust, ignoring any trailing characters such as
            # backslashes or whitespace that may accidentally be included.
            import re  # Local import to avoid global dependency at module level
            numbers = re.findall(r"\d+", sub)
            if not numbers:
                continue
            if len(numbers) >= 2:
                # Treat the first two numbers as a range (inclusive). If the
                # second number is smaller than the first, swap them.
                try:
                    start, end = int(numbers[0]), int(numbers[1])
                except Exception:
                    continue
                if start > end:
                    start, end = end, start
                years.extend(range(start, end + 1))
            elif len(numbers) == 1:
                # Single year
                try:
                    years.append(int(numbers[0]))
                except Exception:
                    continue
    return sorted(set(years))


def parse_utc_local(date_str: str, time_str: str, tz_name: str) -> datetime:
    """Parse MM-DD-YYYY and HH:MM as UTC and convert to target time zone."""
    dt_utc = datetime.strptime(f"{date_str} {time_str}", "%m-%d-%Y %H:%M").replace(tzinfo=timezone.utc)
    if ZoneInfo is None:
        # Fallback: keep UTC if zoneinfo not available
        return dt_utc
    try:
        return dt_utc.astimezone(ZoneInfo(tz_name))
    except Exception:
        return dt_utc  # graceful fallback



# ---- Data Quality Control (QC) helpers ----
def qc_bad_day(reason_map: Dict[datetime.date, str], d: datetime.date, reason: str):
    # keep first reason only to avoid overwriting
    if d not in reason_map:
        reason_map[d] = reason

def qc_filter_days(df_local: pd.DataFrame,
                   min_range_f: float,
                   min_unique: int,
                   max_flat_frac: float,
                   min_samples: int) -> Tuple[pd.DataFrame, List[datetime.date], Dict[datetime.date, str]]:
    """Return a filtered dataframe (still in local time) with bad days removed.
    Criteria:
      - day has < min_samples raw points
      - diurnal range (max-min) < min_range_f
      - number of unique temps < min_unique
      - fraction of identical successive readings > max_flat_frac (flat-lined sensor)
    """
    # Expect columns: [datetime_local, Air Temp (F)]
    if df_local.empty:
        return df_local, [], {}

    # Work on a copy grouped by date
    reasons: Dict[datetime.date, str] = {}
    keep_days: List[datetime.date] = []

    gby = df_local.groupby(df_local["datetime_local"].dt.date)
    for d, g in gby:
        temps = g["Air Temp (F)"].astype(float).to_numpy()
        n = temps.size
        if n < min_samples:
            qc_bad_day(reasons, d, f"too_few_samples:{n}")
            continue
        tmax = float(np.nanmax(temps))
        tmin = float(np.nanmin(temps))
        dr = tmax - tmin
        if not np.isfinite(dr) or dr < min_range_f:
            qc_bad_day(reasons, d, f"low_range:{dr:.2f}")
            continue
        nunique = int(len(np.unique(temps)))
        if nunique < min_unique:
            qc_bad_day(reasons, d, f"low_unique:{nunique}")
            continue
        # flat-line fraction: proportion of successive pairs with exactly no change
        if n > 1:
            diffs = np.diff(temps)
            flat_frac = float(np.mean(np.isclose(diffs, 0.0)))
        else:
            flat_frac = 1.0
        if flat_frac > max_flat_frac:
            qc_bad_day(reasons, d, f"flat_frac:{flat_frac:.2f}")
            continue
        keep_days.append(d)

    if not keep_days:
        return df_local.iloc[0:0].copy(), [], reasons

    mask_keep = df_local["datetime_local"].dt.date.isin(keep_days)
    return df_local.loc[mask_keep].copy(), keep_days, reasons


def resample_to_half_hour(df_local: pd.DataFrame) -> Tuple[np.ndarray, Dict[datetime.date, np.ndarray]]:
    """Build 30-minute daily vectors by interpolating within each day only.
    This avoids global resampling across multi-year gaps, which can fabricate thousands of days.
    """
    df = df_local.sort_values("datetime_local").copy()

    grid = np.arange(0, 24, 0.5)
    daily_obs: Dict[datetime.date, np.ndarray] = {}

    # Group by calendar date and interpolate within the day
    for d, g in df.groupby(df["datetime_local"].dt.date):
        gg = g.sort_values("datetime_local")
        h = gg["datetime_local"].dt.hour.to_numpy() + gg["datetime_local"].dt.minute.to_numpy() / 60.0
        y = gg["Air Temp (F)"].astype(float).to_numpy()

        # Deduplicate minute-times to ensure monotonic x for interpolation
        if h.size > 1:
            gtmp = pd.DataFrame({"h": h, "y": y})
            gtmp["h_round"] = np.round(gtmp["h"] * 60).astype(int)  # minutes since midnight
            gagg = gtmp.groupby("h_round", as_index=False)["y"].mean()
            xs = (gagg["h_round"].to_numpy() / 60.0).astype(float)
            ys = gagg["y"].to_numpy()
        else:
            xs = h
            ys = y

        # Ensure increasing x and drop bad points
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (xs < 24)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size == 0:
            continue

        # Interpolate within-day; outside the in-day hull, hold edge values
        obs_vec = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        daily_obs[d] = obs_vec

    return grid, daily_obs

def generate_case_study(
    *,
    weather_file: str,
    boundary_file: str,
    station: str,
    month: int,
    years: List[int],
    tz_name: str = "America/Los_Angeles",
    risk2_window_hours: float = 2.0,
    risk2_area_thresh: float = 10.0,
    risk_direction: str = "above",
    outdir: str | None = None,
    report_title: str | None = None,
    qc_min_range_f: float = 2.0,
    qc_min_unique: int = 8,
    qc_max_flat_frac: float = 0.80,
    qc_min_samples: int = 24,
) -> CaseStudyOutputs:
    """
    Compute risks over the selected years and month, write CSVs/plots, and build a polished PDF report.

    Parameters
    ----------
    weather_file : path to the weather CSV
    boundary_file : path to the boundary CSV
    station : station code for labeling (e.g. "KEDW")
    month : month (1–12)
    years : list of years to include. All days in the specified month across these years will be combined.
    tz_name : IANA timezone name (e.g., "America/Los_Angeles")
    risk2_window_hours : window length for Risk 2 (hours)
    outdir : output directory (defaults to current working dir)
    report_title : custom report title (optional)

    Returns
    -------
    CaseStudyOutputs : dataclass containing paths to outputs and a stats dict.
    """
    if not years:
        raise ValueError("No years specified for analysis.")

    outdir = outdir or os.getcwd()
    os.makedirs(outdir, exist_ok=True)

    # Load inputs
    df_wx = pd.read_csv(weather_file)
    df_bnd = pd.read_csv(boundary_file)

    # Basic schema checks
    for col in ("Date", "Time (UTC)", "Air Temp (F)"):
        if col not in df_wx.columns:
            raise ValueError(f"Weather file missing required column: {col}")
    for col in ("hour", "temp"):
        if col not in df_bnd.columns:
            raise ValueError(f"Boundary file missing required column: {col}")

    # Convert to local time and filter month/year
    df_wx["datetime_local"] = df_wx.apply(
        lambda r: parse_utc_local(r["Date"], r["Time (UTC)"], tz_name), axis=1
    )
    # Build masks explicitly to avoid any operator‑precedence surprises
    year_mask = df_wx["datetime_local"].dt.year.isin(years)
    month_mask = df_wx["datetime_local"].dt.month == month
    mask = year_mask & month_mask
    df_mon = df_wx.loc[mask, ["datetime_local", "Air Temp (F)"]].copy()

    # Sanity check: ensure we only captured the requested month
    months_present = sorted(df_mon["datetime_local"].dt.month.unique().tolist())
    if len(months_present) != 1 or months_present[0] != month:
        raise ValueError(
            "Month filter failed: requested month={} but found months={}. "
            "Please report this with your command line so we can reproduce.".format(month, months_present)
        )

    if df_mon.empty:
        raise ValueError(
            "No data found for the specified month/year(s) after timezone conversion. "
            f"month={month}, years={years}, tz={tz_name}"
        )

    print(f"[DEBUG] Filtered records: {len(df_mon):,} | Days: {df_mon['datetime_local'].dt.date.nunique():,} | Years: {sorted(set(df_mon['datetime_local'].dt.year))} | Month: {month}")

    # --- Data Quality: remove flat-line / low-variance / sparse days
    df_mon, kept_days, qc_reasons = qc_filter_days(
        df_mon,
        min_range_f=qc_min_range_f,
        min_unique=qc_min_unique,
        max_flat_frac=qc_max_flat_frac,
        min_samples=qc_min_samples,
    )
    n_days_qc = int(len(kept_days))
    if n_days_qc == 0:
        raise ValueError("All days failed QC; relax QC thresholds or check data quality.")

    # Use only the month-filtered dataset for counts/labels
    present_years = sorted(df_mon["datetime_local"].dt.year.unique().tolist())
    n_days_mon = int(df_mon["datetime_local"].dt.date.nunique())
    # Build 30‑min daily vectors for this month only
    grid, daily_obs = resample_to_half_hour(df_mon)
    # n_days_qc is the actual number of days after QC

    # Boundary on same grid
    bnd_vec = np.interp(grid, df_bnd["hour"].values, df_bnd["temp"].values)
    bnd_peak_idx = int(np.nanargmax(bnd_vec))
    bnd_peak_temp = float(np.nanmax(bnd_vec))

    # Direction-aware settings
    is_above = (risk_direction == "above")
    # For Risk 1, compare to boundary peak (hot) or trough (cold)
    bnd_trough_idx = int(np.nanargmin(bnd_vec))
    bnd_trough_temp = float(np.nanmin(bnd_vec))
    r1_threshold = bnd_peak_temp if is_above else bnd_trough_temp

    dates = sorted(daily_obs.keys())
    n_eval = len(dates)
    if n_eval != n_days_qc:
        print(f"[WARN] Unique dates from resample/group ({n_eval}) != calendar day count after QC ({n_days_qc}). Using n_eval={n_eval} as denominator for risk stats.")

    # ---- Risk 1: daily peak vs boundary peak (no shift)
    k1 = 0
    for d in dates:
        if is_above:
            obs_ext = float(np.nanmax(daily_obs[d]))
            exceed = obs_ext > r1_threshold
        else:
            obs_ext = float(np.nanmin(daily_obs[d]))
            exceed = obs_ext < r1_threshold
        if exceed:
            k1 += 1
    (ci1_low, ci1_high), p1 = wilson_ci(k1, n_eval)

    # ---- Risk 2: any contiguous N‑hour window above boundary after peak‑alignment
    win_steps = int(round(risk2_window_hours / 0.5))
    k2 = 0
    degree_hours: List[float] = []

    # Examples for plotting
    ex_day_r1: datetime.date | None = None
    nx_day_r1: datetime.date | None = None
    ex_day_r2: datetime.date | None = None
    nx_day_r2: datetime.date | None = None

    # For stacked plot
    r1_exceed_dates: List[datetime.date] = []
    r2_exceed_dates: List[datetime.date] = []

    for d in dates:
        obs_vec = daily_obs[d]
        # R1 examples
        if is_above:
            obs_ext = float(np.nanmax(obs_vec))
            if ex_day_r1 is None and obs_ext > r1_threshold:
                ex_day_r1 = d
            if nx_day_r1 is None and obs_ext <= r1_threshold:
                nx_day_r1 = d
        else:
            obs_ext = float(np.nanmin(obs_vec))
            if ex_day_r1 is None and obs_ext < r1_threshold:
                ex_day_r1 = d
            if nx_day_r1 is None and obs_ext >= r1_threshold:
                nx_day_r1 = d

        # Align extremes: peak for hot (above), trough for cold (below)
        if is_above:
            obs_ext_idx = int(np.nanargmax(obs_vec))
            step = obs_ext_idx - bnd_peak_idx
        else:
            obs_ext_idx = int(np.nanargmin(obs_vec))
            step = obs_ext_idx - bnd_trough_idx
        bnd_shifted = np.roll(bnd_vec, step)
        # Margin is positive when the day violates the boundary criterion
        margin = (obs_vec - bnd_shifted) if is_above else (bnd_shifted - obs_vec)

        # degree‑hours above/below boundary
        degree_hours.append(float(np.sum(np.clip(margin, 0, None)) * 0.5))

        # contiguous window check
        exceed_win = any(
            np.all(margin[i:i + win_steps] > 0) for i in range(0, len(margin) - win_steps + 1)
        )
        if exceed_win:
            k2 += 1
        if ex_day_r2 is None and exceed_win:
            ex_day_r2 = d
        if nx_day_r2 is None and not exceed_win:
            nx_day_r2 = d

        if exceed_win:
            r2_exceed_dates.append(d)
        # For R1 stacked, add days that exceed the threshold
        if (is_above and float(np.nanmax(obs_vec)) > r1_threshold) or (not is_above and float(np.nanmin(obs_vec)) < r1_threshold):
            r1_exceed_dates.append(d)

    (ci2_low, ci2_high), p2 = wilson_ci(k2, n_eval)

    # Area-based Risk 2: thermal load exceedance using degree-hours threshold
    degree_hours_arr = np.array(degree_hours, dtype=float)
    k2_area = int(np.sum(degree_hours_arr > risk2_area_thresh))
    (ci2a_low, ci2a_high), p2_area = wilson_ci(k2_area, n_eval)

    # ---- File name helpers
    # Year label based on data actually present after filtering
    sorted_years = present_years
    if len(sorted_years) == 1:
        years_label = str(sorted_years[0])
    else:
        contiguous = all(sorted_years[i] + 1 == sorted_years[i + 1] for i in range(len(sorted_years) - 1))
        years_label = f"{sorted_years[0]}-{sorted_years[-1]}" if contiguous else "_".join(str(y) for y in sorted_years)

    stem = f"{station}_{years_label}-{month:02d}"

    def path(name: str) -> str:
        return os.path.join(outdir, f"{stem}_{name}")

    # ---- CSVs
    # Risk 1 day results
    r1_rows = []
    for d in dates:
        obs_peak = float(np.nanmax(daily_obs[d]))
        obs_trough = float(np.nanmin(daily_obs[d]))
        if is_above:
            obs_ext = obs_peak
            exceed = obs_ext > r1_threshold
        else:
            obs_ext = obs_trough
            exceed = obs_ext < r1_threshold
        r1_rows.append({
            "date": d,
            "obs_peak_F": obs_peak,
            "obs_trough_F": obs_trough,
            "bnd_peak_F": bnd_peak_temp,
            "bnd_trough_F": bnd_trough_temp,
            "threshold_F": r1_threshold,
            "exceed": exceed,
        })
    df_r1 = pd.DataFrame(r1_rows)
    csv_risk1 = path("risk1_daily_peaks.csv")
    df_r1.to_csv(csv_risk1, index=False)

    # Risk 2 day results
    r2_rows: List[Dict] = []
    for idx, d in enumerate(dates):
        obs_vec = daily_obs[d]
        # Align extremes as above
        if is_above:
            obs_ext_idx = int(np.nanargmax(obs_vec))
            step = obs_ext_idx - bnd_peak_idx
        else:
            obs_ext_idx = int(np.nanargmin(obs_vec))
            step = obs_ext_idx - bnd_trough_idx
        bnd_shifted = np.roll(bnd_vec, step)
        margin = (obs_vec - bnd_shifted) if is_above else (bnd_shifted - obs_vec)
        degree_hrs = float(np.sum(np.clip(margin, 0, None)) * 0.5)
        exceed_win = any(
            np.all(margin[i:i + win_steps] > 0) for i in range(0, len(margin) - win_steps + 1)
        )
        best_mean = -float('inf')
        best_start = float('nan')
        if exceed_win:
            for i in range(0, len(margin) - win_steps + 1):
                w = margin[i:i + win_steps]
                if np.all(w > 0):
                    m = float(np.mean(w))
                    if m > best_mean:
                        best_mean = m
                        best_start = float(grid[i])
        r2_rows.append({
            "date": d,
            "exceed_window": exceed_win,
            "best_window_start_local_hr": best_start if exceed_win else float('nan'),
            "best_window_mean_margin_F": best_mean if exceed_win else 0.0,
            "degree_hours_beyond_boundary": degree_hrs,
            "risk2_window_hours": risk2_window_hours,
            "exceed_area_threshold": bool(degree_hrs > risk2_area_thresh),
        })
    df_r2 = pd.DataFrame(r2_rows)
    csv_risk2 = path(f"risk2_{int(risk2_window_hours)}h.csv")
    df_r2.to_csv(csv_risk2, index=False)

    # ---- QC dropped-days CSV export
    qc_csv = path("qc_dropped_days.csv")
    if qc_reasons:
        pd.DataFrame([
            {"date": d, "reason": qc_reasons[d]} for d in sorted(qc_reasons)
        ]).to_csv(qc_csv, index=False)
    else:
        # create an empty file with header to be explicit
        pd.DataFrame(columns=["date", "reason"]).to_csv(qc_csv, index=False)

    # ---- Plots
    # Risk 1 examples
    plot_risk1_examples: str | None = None
    if ex_day_r1 is not None and nx_day_r1 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        grid_hours = grid
        for ax, day, title in zip(axes, [ex_day_r1, nx_day_r1], ["Exceedance Example", "Non-exceedance Example"]):
            obs = daily_obs[day]
            ax.plot(grid_hours, obs, linewidth=2, label=f"Observed {day}")
            ax.axhline(r1_threshold, color="red", linestyle="--", label=("Boundary peak temp" if is_above else "Boundary trough temp"))
            ax.set_title(title)
            ax.set_xlabel("Local Hour")
            ax.set_ylabel("Temp (°F)")
            ax.grid(True)
            ax.legend()
        plot_risk1_examples = path("risk1_examples.png")
        plt.tight_layout()
        plt.savefig(plot_risk1_examples)
        plt.close()

    # Risk 1 stacked
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in r1_exceed_dates:
        ax.plot(grid, daily_obs[d], alpha=0.5)
    ax.axhline(r1_threshold, color="red", linestyle="--", linewidth=2, label=("Boundary peak temp" if is_above else "Boundary trough temp"))
    ax.set_title(f"Risk 1: Observed Curves on Exceedance Days ({station} {years_label}-{month:02d})")
    ax.set_xlabel("Local Hour")
    ax.set_ylabel("Temp (°F)")
    ax.legend()
    ax.grid(True)
    plot_risk1_stacked = path("risk1_stacked.png")
    plt.tight_layout()
    plt.savefig(plot_risk1_stacked)
    plt.close()

    # Risk 2 examples
    plot_risk2_examples: str | None = None
    if ex_day_r2 is not None and nx_day_r2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, day, title in zip(axes, [ex_day_r2, nx_day_r2], ["Exceedance Example", "Non-exceedance Example"]):
            obs = daily_obs[day]
            if is_above:
                obs_ext_idx = int(np.nanargmax(obs))
                step = obs_ext_idx - bnd_peak_idx
            else:
                obs_ext_idx = int(np.nanargmin(obs))
                step = obs_ext_idx - bnd_trough_idx
            bnd_shifted = np.roll(bnd_vec, step)
            margin = (obs - bnd_shifted) if is_above else (bnd_shifted - obs)
            ax.plot(grid, obs, linewidth=2, label=f"Observed {day}")
            ax.plot(grid, bnd_shifted, linewidth=2, label="Shifted Boundary", color="red")
            ax.fill_between(grid, obs, bnd_shifted, where=(margin > 0), alpha=0.35)
            ax.set_title(title)
            ax.set_xlabel("Local Hour")
            ax.set_ylabel("Temp (°F)")
            ax.grid(True)
            ax.legend()
        plot_risk2_examples = path("risk2_examples.png")
        plt.tight_layout()
        plt.savefig(plot_risk2_examples)
        plt.close()

    # Risk 2 stacked (all 2h exceedance days)
    plot_risk2_stacked: str | None = None
    if len(r2_exceed_dates) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for d in r2_exceed_dates:
            ax.plot(grid, daily_obs[d], alpha=0.5)
        # Overlay a single (unshifted) boundary curve for visual reference
        ax.plot(grid, bnd_vec, linewidth=2, color="red", label="Boundary curve (Example)")
        ax.legend()
        ax.set_title(f"Risk 2: Observed Curves on {int(risk2_window_hours)}h Exceedance Days ({'above' if is_above else 'below'} boundary) ({station} {years_label}-{month:02d})")
        ax.set_xlabel("Local Hour")
        ax.set_ylabel("Temp (°F)")
        ax.grid(True)
        plot_risk2_stacked = path("risk2_stacked.png")
        plt.tight_layout()
        plt.savefig(plot_risk2_stacked)
        plt.close()

    # Severity histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degree_hours, bins=12, edgecolor="black")
    if risk2_area_thresh and risk2_area_thresh > 0:
        ax.axvline(risk2_area_thresh, linestyle="--", linewidth=2, label=f"Threshold {risk2_area_thresh:.1f} °F·h")
        ax.legend()
    ax.set_xlabel("Degree-hours beyond boundary (°F·h)")
    ax.set_ylabel("Days")
    ax.set_title(f"Thermal Loading Severity ({station} {years_label}-{month:02d})")
    plot_severity_hist = path("severity_hist.png")
    plt.tight_layout()
    plt.savefig(plot_severity_hist)
    plt.close()

    # ---- PDF
    if not _HAVE_REPORTLAB:
        pdf_path = path("CaseStudy_NEEDS_REPORTLAB.txt")
        with open(pdf_path, "w", encoding="utf-8") as fh:
            fh.write("Install reportlab to enable PDF generation: pip install reportlab\n")
    else:
        if not report_title:
            # Compose a default title: station, month name, years range
            month_name = calendar.month_name[month]
            period_desc = f"{month_name} {years_label}"
            report_title = (
                f"Case Study: {station} {period_desc} — Temperature Risks vs Certification Boundary"
            )
        pdf_path = path("CaseStudy.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story: List = []
        story.append(Paragraph(f"<b>{report_title}</b>", styles["Title"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Results & Probability Statements</b>", styles["Heading2"]))
        month_name = calendar.month_name[month]
        period_desc = f"{month_name} {years_label}"
        # --- QC summary in PDF ---
        dropped = len(qc_reasons)
        if dropped:
            story.append(Paragraph(
                f"Data quality screen removed <b>{dropped}</b> day(s) for issues like low diurnal range, too few samples, or flat-line readings. "
                f"Probability estimates below use the remaining <b>{n_days_qc}</b> day(s). See qc_dropped_days.csv for details.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 12))
        story.append(
            Paragraph(
                (
                    (f"• <b>Risk 1 (Peak exceedance)</b>: {k1} of {n_eval} days exceeded the boundary peak. "
                     f"The estimated probability that a randomly selected day from the selected period ({period_desc}) exceeds the boundary peak is "
                     f"<b>{p1 * 100:.1f}%</b>. "
                     f"With 95% confidence, the true probability lies between <b>{ci1_low * 100:.1f}%</b> and <b>{ci1_high * 100:.1f}%</b>.")
                    if is_above else
                    (f"• <b>Risk 1 (Trough exceedance)</b>: {k1} of {n_eval} days fell below the boundary trough. "
                     f"The estimated probability that a randomly selected day from the selected period ({period_desc}) falls below the boundary trough is "
                     f"<b>{p1 * 100:.1f}%</b>. "
                     f"With 95% confidence, the true probability lies between <b>{ci1_low * 100:.1f}%</b> and <b>{ci1_high * 100:.1f}%</b>.")
                ),
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                (
                    (f"• <b>Risk 2 (Thermal loading, {risk2_window_hours:.0f} h)</b>: {k2} of {n_eval} days contained a continuous {risk2_window_hours:.0f}-hour period with observed temperature above the boundary (after peak alignment). "
                     f"Estimated probability: <b>{p2 * 100:.1f}%</b> (95% CI <b>{ci2_low * 100:.1f}%</b>–<b>{ci2_high * 100:.1f}%</b>).")
                    if is_above else
                    (f"• <b>Risk 2 (Thermal loading, {risk2_window_hours:.0f} h)</b>: {k2} of {n_eval} days contained a continuous {risk2_window_hours:.0f}-hour period with observed temperature below the boundary (after trough alignment). "
                     f"Estimated probability: <b>{p2 * 100:.1f}%</b> (95% CI <b>{ci2_low * 100:.1f}%</b>–<b>{ci2_high * 100:.1f}%</b>).")
                ),
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                (
                    (f"• <b>Risk 2 (Thermal load, area)</b>: {k2_area} of {n_eval} days had total positive degree-hours above the boundary exceeding <b>{risk2_area_thresh:.1f} °F·h</b> (after peak alignment). "
                     f"Estimated probability over {period_desc}: <b>{p2_area * 100:.1f}%</b> (95% CI <b>{ci2a_low * 100:.1f}%</b>–<b>{ci2a_high * 100:.1f}%</b>).")
                    if is_above else
                    (f"• <b>Risk 2 (Thermal load, area)</b>: {k2_area} of {n_eval} days had total positive degree-hours below the boundary exceeding <b>{risk2_area_thresh:.1f} °F·h</b> (after trough alignment). "
                     f"Estimated probability over {period_desc}: <b>{p2_area * 100:.1f}%</b> (95% CI <b>{ci2a_low * 100:.1f}%</b>–<b>{ci2a_high * 100:.1f}%</b>).")
                ),
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 12))
        story.append(Image(plot_severity_hist, width=400, height=250))
        story.append(Spacer(1, 12))
        
        story.append(Spacer(1, 12))
        add_plot_or_text(story, plot_risk1_examples, "Risk 1 (examples)")
        add_plot_or_text(story, plot_risk1_stacked, "Risk 1 (stacked)")
        add_plot_or_text(story, plot_risk2_examples, "Risk 2 (examples)")
        add_plot_or_text(story, plot_risk2_stacked, "Risk 2 (stacked)")
        # --- Add Risk 2 area examples plot to PDF ---
        # For this, we need to reconstruct a dataframe with the appropriate columns.
        # The inputs: daily_obs (dict of date -> 48-vector), bnd_vec (boundary on grid)
        # We'll use the same peak alignment as for risk2, for consistency.
        # Compose a DataFrame for plotting examples
        df_area_plot = []
        for d in dates:
            obs_vec = daily_obs[d]
            obs_peak_idx = int(np.nanargmax(obs_vec))
            step = obs_peak_idx - bnd_peak_idx
            bnd_shifted = np.roll(bnd_vec, step)
            # For area, we consider only the 24-hourly values for visual clarity
            # Interpolate to hourly grid
            hour_grid = np.arange(24)
            obs_hourly = np.interp(hour_grid, grid, obs_vec)
            bnd_hourly = np.interp(hour_grid, grid, bnd_shifted)
            df_area_plot.append(
                {
                    "date": d,
                    "temp": obs_hourly,
                    "shifted_boundary": bnd_hourly,
                }
            )
        # Build a DataFrame with multi-index: (date, hour)
        df_area = []
        for row in df_area_plot:
            for h in range(24):
                df_area.append(
                    {
                        "date": row["date"],
                        "hour": h,
                        "temp": row["temp"][h],
                        "shifted_boundary": row["shifted_boundary"][h],
                    }
                )
        df_area = pd.DataFrame(df_area)
        # Pivot to get one day's temps as a vector (needed for plotting function)
        # For each day, get the 24-hour vector
        day_groups = {}
        for d, g in df_area.groupby("date"):
            arr = np.array([g.loc[g["hour"] == h, "temp"].values[0] for h in range(24)])
            bnd = np.array([g.loc[g["hour"] == h, "shifted_boundary"].values[0] for h in range(24)])
            day_groups[d] = (arr, bnd)
        # For plotting, we need a DataFrame with index as datetime (date), column "temp"
        # We'll pick the first shifted_boundary (since they're all aligned for each day).
        # We'll pass the DataFrame with index as datetime, column "temp", and the shifted_boundary as a vector
        # Use the first day's boundary as example
        if len(day_groups) > 0:
            # Build df for plotting
            rows = []
            for d, (obs, bnd) in day_groups.items():
                for h in range(24):
                    rows.append({"datetime": pd.Timestamp(d) + pd.Timedelta(hours=h), "temp": obs[h]})
            df_plot = pd.DataFrame(rows).set_index("datetime")
            # Use the first day's shifted boundary for plotting (they are all aligned)
            shifted_boundary = list(day_groups.values())[0][1]
            risk2_area_examples_file = plot_risk2_area_examples(
                df_plot, shifted_boundary, station, month, outdir, theta_area=risk2_area_thresh, direction=risk_direction
            )
            add_plot_or_text(story, str(risk2_area_examples_file), "Risk 2 (Area Examples)")

        doc.build(story)

    return CaseStudyOutputs(
        pdf=pdf_path,
        csv_risk1=csv_risk1,
        csv_risk2=csv_risk2,
        plot_risk1_examples=plot_risk1_examples,
        plot_risk1_stacked=plot_risk1_stacked,
        plot_risk2_examples=plot_risk2_examples,
        plot_risk2_stacked=plot_risk2_stacked,
        plot_severity_hist=plot_severity_hist,
        stats={
            "n_days": n_eval,
            "years": years,
            "month": month,
            "risk_direction": risk_direction,
            "r1_threshold_F": r1_threshold,
            "risk1": {
                "k": k1,
                "p": p1,
                "ci": (ci1_low, ci1_high),
                "bnd_peak_F": bnd_peak_temp,
            },
            "risk2": {
                "k": k2,
                "p": p2,
                "ci": (ci2_low, ci2_high),
                "window_h": risk2_window_hours,
            },
            "risk2_area": {
                "k": k2_area,
                "p": p2_area,
                "ci": (ci2a_low, ci2a_high),
                "threshold_Fh": risk2_area_thresh,
            },
        },
    )


def parse_months_input(tokens: List[str] | None) -> List[int]:
    """Parse months like '7', '4-9', or '1,2,12' (wrap-around ranges allowed)."""
    if not tokens:
        return []
    months: set[int] = set()
    for tok in tokens:
        if tok is None:
            continue
        for part in str(tok).split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                a = a.strip(); b = b.strip()
                if a.isdigit() and b.isdigit():
                    a = int(a); b = int(b)
                    if 1 <= a <= 12 and 1 <= b <= 12:
                        if a <= b:
                            rng = range(a, b+1)
                        else:
                            rng = list(range(a, 13)) + list(range(1, b+1))
                        months.update(rng)
            else:
                if part.isdigit():
                    m = int(part)
                    if 1 <= m <= 12:
                        months.add(m)
    return sorted(months)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate temperature vs boundary case studies.")
    parser.add_argument("--weather", required=True, help="Path to weather CSV")
    parser.add_argument(
        "--boundary",
        required=True,
        help="Boundary CSV file with columns: hour,temp",
    )
    parser.add_argument("--station", required=True, help="Station code for labeling (e.g., KEDW)")
    parser.add_argument(
        "--month", type=int, action="append", required=True, help="Month 1-12 (repeatable)"
    )
    # Support both --years (single argument) and deprecated --year (repeatable)
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help=(
            "Comma-separated years and/or ranges for aggregated analysis (e.g. '2015-2025' or '2007' or '2015,2017,2020'). "
            "If provided, all days in the specified month(s) across these years will be combined."
        ),
    )
    parser.add_argument(
        "--year",
        action="append",
        default=None,
        help=(
            "Year or year range (repeatable). Deprecated; use --years instead. "
            "Each entry may be a single year (e.g. '2010') or range (e.g. '2015-2018'). "
            "All days across the specified years will be aggregated."
        ),
    )
    parser.add_argument(
        "--tz",
        default="America/Los_Angeles",
        help="IANA timezone (e.g., America/Los_Angeles)",
    )
    parser.add_argument(
        "--risk2-hours",
        type=float,
        default=2.0,
        dest="risk2_hours",
        help="Window length for Risk 2 (hours)",
    )
    parser.add_argument(
        "--risk2-area-thresh",
        type=float,
        default=10.0,
        help="Area-based thermal load threshold in degree-hours (°F·h) for Risk 2 (area). Default: 10.0",
    )
    parser.add_argument(
        "--risk-direction",
        choices=["above", "below"],
        default="above",
        help="Direction of exceedance: 'above' (hot risk) or 'below' (cold risk). Default: above",
    )
    # QC CLI options
    parser.add_argument("--qc-min-range-f", type=float, default=2.0,
                        help="Minimum daily diurnal range (°F) required; days with smaller range are dropped (default: 2°F)")
    parser.add_argument("--qc-min-unique", type=int, default=8,
                        help="Minimum unique temperature readings per day; days with fewer are dropped (default: 8)")
    parser.add_argument("--qc-max-flat-frac", type=float, default=0.80,
                        help="Max fraction of identical successive readings allowed in a day; above this the day is dropped (default: 0.80)")
    parser.add_argument("--qc-min-samples", type=int, default=24,
                        help="Minimum number of samples in a day before resampling; days with fewer are dropped (default: 24)")
    parser.add_argument(
        "--outdir", default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom report title (optional). If not provided, a default based on station, month, and years is used.",
    )

    args = parser.parse_args()

    # Parse years
    year_tokens: List[str] = []
    if args.years:
        year_tokens.append(args.years)
    if args.year:
        year_tokens.extend(args.year)

    years: List[int] = parse_years_input(year_tokens)
    if not years:
        raise SystemExit("At least one year must be specified via --years or --year.")

    # Run case study for each month with aggregated years
    for m in args.month:
        # Compose a title if not provided, else reuse the user's title
        title = args.title
        outputs = generate_case_study(
            weather_file=args.weather,
            boundary_file=args.boundary,
            station=args.station,
            month=m,
            years=years,
            tz_name=args.tz,
            risk2_window_hours=args.risk2_hours,
            risk2_area_thresh=args.risk2_area_thresh,
            risk_direction=args.risk_direction,
            outdir=args.outdir,
            report_title=title,
            qc_min_range_f=args.qc_min_range_f,
            qc_min_unique=args.qc_min_unique,
            qc_max_flat_frac=args.qc_max_flat_frac,
            qc_min_samples=args.qc_min_samples,
        )
        print("\n=== Generated Case Study ===")
        print("PDF:", outputs.pdf)
        print("Risk1 CSV:", outputs.csv_risk1)
        print("Risk2 CSV:", outputs.csv_risk2)
        print("Risk1 examples:", outputs.plot_risk1_examples)
        print("Risk1 stacked:", outputs.plot_risk1_stacked)
        print("Risk2 examples:", outputs.plot_risk2_examples)
        print("Risk2 stacked:", outputs.plot_risk2_stacked)
        print("Severity histogram:", outputs.plot_severity_hist)
        print("Stats:", outputs.stats)
        print("Risk2 area threshold (°F·h):", args.risk2_area_thresh)
        print("Risk direction:", args.risk_direction)
        # Print QC dropped-days CSV path
        print("QC dropped-days CSV:", os.path.join(args.outdir, f"{outputs.stats['years'][0]}-{outputs.stats['month']:02d}_qc_dropped_days.csv") if 'years' in outputs.stats else '(see output directory)')


if __name__ == "__main__":
    main()