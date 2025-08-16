from __future__ import annotations

from reportlab.platypus import Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

# Helper: Add plot image to PDF story if available, else fallback text.
def add_plot_or_text(story, plot_path, description, width=400, height=250):
    """
    Add a plot image to the PDF story if available, otherwise add fallback text.
    
    Args:
        story (list): The PDF story flowables list.
        plot_path (str or None): Path to the plot image, or None if not generated.
        description (str): Text to display if no plot is available.
        width (int): Width of the image in the PDF.
        height (int): Height of the image in the PDF.
    """
    if plot_path:
        story.append(Image(plot_path, width=width, height=height))
    else:
        story.append(Paragraph(f"<b>No exceedance events found for {description}.</b>", styles["Normal"]))
    story.append(Spacer(1, 12))
from pathlib import Path
def plot_risk2_area_examples(df, shifted_boundary, station, month, outdir, theta_area):
    """
    Plot example days showing exceedance and non-exceedance for area-under-curve thermal load.
    """
    local_hours = np.arange(24)
    exceed_days = []
    non_exceed_days = []

    # Compute area exceedance for each day
    for date, group in df.groupby(df.index.date):
        observed = group["temp"].values
        area = np.trapz(np.clip(observed - shifted_boundary, 0, None), local_hours)
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
        axes[0].fill_between(local_hours, shifted_boundary, observed,
                             where=observed > shifted_boundary, color="blue", alpha=0.3)
        axes[0].set_title("Exceedance Example")
        axes[0].legend()
        axes[0].annotate(f"A⁺ = {area:.1f} °F·h\nθ = {theta_area} °F·h",
                         xy=(0.05, 0.9), xycoords="axes fraction", fontsize=10,
                         bbox=dict(facecolor="white", alpha=0.7))

    if non_exceed_days:
        date, observed, area = non_exceed_days[0]
        axes[1].plot(local_hours, observed, label=f"Observed {date}", color="blue")
        axes[1].plot(local_hours, shifted_boundary, label="Shifted Boundary", color="red")
        axes[1].fill_between(local_hours, shifted_boundary, observed,
                             where=observed > shifted_boundary, color="blue", alpha=0.3)
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
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
    _HAVE_REPORTLAB = True
except Exception:
    _HAVE_REPORTLAB = False

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


def resample_to_half_hour(df_local: pd.DataFrame) -> Tuple[np.ndarray, Dict[datetime.date, np.ndarray]]:
    """Resample to a 30‑minute grid and return per‑day vectors on 0.5 h grid."""
    df = df_local.sort_values("datetime_local").set_index("datetime_local")
    df_30 = df.resample("30min").mean(numeric_only=True)
    # time interpolation, fill both directions to avoid NaNs
    df_30["Air Temp (F)"] = df_30["Air Temp (F)"].interpolate(method="time", limit_direction="both")
    df_30["h_of_day"] = df_30.index.hour + df_30.index.minute / 60.0

    grid = np.arange(0, 24, 0.5)
    daily_obs: Dict[datetime.date, np.ndarray] = {}
    # Group by the actual calendar day derived from the DateTimeIndex to avoid any dtype surprises
    for d, g in df_30.groupby(df_30.index.date):
        obs_vec = np.interp(grid, g["h_of_day"].values, g["Air Temp (F)"].values)
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
    outdir: str | None = None,
    report_title: str | None = None,
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

    # Use only the month-filtered dataset for counts/labels
    present_years = sorted(df_mon["datetime_local"].dt.year.unique().tolist())
    n_days_mon = int(df_mon["datetime_local"].dt.date.nunique())
    # Build 30‑min daily vectors for this month only
    grid, daily_obs = resample_to_half_hour(df_mon)

    # Boundary on same grid
    bnd_vec = np.interp(grid, df_bnd["hour"].values, df_bnd["temp"].values)
    bnd_peak_idx = int(np.nanargmax(bnd_vec))
    bnd_peak_temp = float(np.nanmax(bnd_vec))

    dates = sorted(daily_obs.keys())
    if len(dates) != n_days_mon:
        print(f"[WARN] Unique dates from resample/group ({len(dates)}) != calendar day count ({n_days_mon}). Proceeding with grouped days.")

    # ---- Risk 1: daily peak vs boundary peak (no shift)
    k1 = 0
    for d in dates:
        obs_peak = float(np.nanmax(daily_obs[d]))
        if obs_peak > bnd_peak_temp:
            k1 += 1
    (ci1_low, ci1_high), p1 = wilson_ci(k1, n_days_mon)

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
        if ex_day_r1 is None and float(np.nanmax(obs_vec)) > bnd_peak_temp:
            ex_day_r1 = d
        if nx_day_r1 is None and float(np.nanmax(obs_vec)) <= bnd_peak_temp:
            nx_day_r1 = d

        # Peak alignment for R2
        obs_peak_idx = int(np.nanargmax(obs_vec))
        step = obs_peak_idx - bnd_peak_idx
        bnd_shifted = np.roll(bnd_vec, step)
        margin = obs_vec - bnd_shifted

        # degree‑hours above boundary
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
        if float(np.nanmax(obs_vec)) > bnd_peak_temp:
            r1_exceed_dates.append(d)

    (ci2_low, ci2_high), p2 = wilson_ci(k2, n_days_mon)

    # Area-based Risk 2: thermal load exceedance using degree-hours threshold
    degree_hours_arr = np.array(degree_hours, dtype=float)
    k2_area = int(np.sum(degree_hours_arr > risk2_area_thresh))
    (ci2a_low, ci2a_high), p2_area = wilson_ci(k2_area, n_days_mon)

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
    r1_rows = [
        {
            "date": d,
            "obs_peak_F": float(np.nanmax(daily_obs[d])),
            "bnd_peak_F": bnd_peak_temp,
            "exceed": float(np.nanmax(daily_obs[d])) > bnd_peak_temp,
        }
        for d in dates
    ]
    df_r1 = pd.DataFrame(r1_rows)
    csv_risk1 = path("risk1_daily_peaks.csv")
    df_r1.to_csv(csv_risk1, index=False)

    # Risk 2 day results
    r2_rows: List[Dict] = []
    for idx, d in enumerate(dates):
        obs_vec = daily_obs[d]
        obs_peak_idx = int(np.nanargmax(obs_vec))
        step = obs_peak_idx - bnd_peak_idx
        bnd_shifted = np.roll(bnd_vec, step)
        margin = obs_vec - bnd_shifted
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
            "degree_hours_above_boundary": degree_hrs,
            "risk2_window_hours": risk2_window_hours,
            "exceed_area_threshold": bool(degree_hrs > risk2_area_thresh),
        })
    df_r2 = pd.DataFrame(r2_rows)
    csv_risk2 = path(f"risk2_{int(risk2_window_hours)}h.csv")
    df_r2.to_csv(csv_risk2, index=False)

    # ---- Plots
    # Risk 1 examples
    plot_risk1_examples: str | None = None
    if ex_day_r1 is not None and nx_day_r1 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        grid_hours = grid
        for ax, day, title in zip(axes, [ex_day_r1, nx_day_r1], ["Exceedance Example", "Non-exceedance Example"]):
            obs = daily_obs[day]
            ax.plot(grid_hours, obs, linewidth=2, label=f"Observed {day}")
            ax.axhline(bnd_peak_temp, color="red", linestyle="--", label="Boundary peak temp")
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
    ax.axhline(bnd_peak_temp, color="red", linestyle="--", linewidth=2, label="Boundary peak temp")
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
            obs_peak_idx = int(np.nanargmax(obs))
            step = obs_peak_idx - bnd_peak_idx
            bnd_shifted = np.roll(bnd_vec, step)
            margin = obs - bnd_shifted
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
        ax.set_title(f"Risk 2: Observed Curves on 2h Exceedance Days ({station} {years_label}-{month:02d})")
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
    ax.set_xlabel("Degree-hours above boundary (°F·h)")
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
        styles = getSampleStyleSheet()
        story: List = []
        story.append(Paragraph(f"<b>{report_title}</b>", styles["Title"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Results & Probability Statements</b>", styles["Heading2"]))
        month_name = calendar.month_name[month]
        period_desc = f"{month_name} {years_label}"
        story.append(
            Paragraph(
                (
                    f"• <b>Risk 1 (Peak exceedance)</b>: {k1} of {n_days_mon} days exceeded the boundary peak. "
                    f"The estimated probability that a randomly selected day from the selected period ({period_desc}) exceeds the boundary peak is "
                    f"<b>{p1 * 100:.1f}%</b>. "
                    f"With 95% confidence, the true probability lies between <b>{ci1_low * 100:.1f}%</b> and <b>{ci1_high * 100:.1f}%</b>."
                ),
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                (
                    f"• <b>Risk 2 (Thermal loading, {risk2_window_hours:.0f} h)</b>: {k2} of {n_days_mon} days contained a continuous {risk2_window_hours:.0f}-hour exceedance after peak alignment. "
                    f"The estimated probability that a randomly selected day from the selected period ({period_desc}) exceeds the thermal loading criterion is "
                    f"<b>{p2 * 100:.1f}%</b>. "
                    f"With 95% confidence, the true probability lies between <b>{ci2_low * 100:.1f}%</b> and <b>{ci2_high * 100:.1f}%</b>."
                ),
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                (
                    f"• <b>Risk 2 (Thermal load, area)</b>: {k2_area} of {n_days_mon} days had total positive degree-hours above the boundary exceeding <b>{risk2_area_thresh:.1f} °F·h</b> (after peak alignment). "
                    f"The estimated probability over {period_desc} is <b>{p2_area * 100:.1f}%</b>. "
                    f"With 95% confidence, the true probability lies between <b>{ci2a_low * 100:.1f}%</b> and <b>{ci2a_high * 100:.1f}%</b>."
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
                df_plot, shifted_boundary, station, month, outdir, theta_area=risk2_area_thresh
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
            "n_days": n_days_mon,
            "years": years,
            "month": month,
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
            outdir=args.outdir,
            report_title=title,
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


if __name__ == "__main__":
    main()