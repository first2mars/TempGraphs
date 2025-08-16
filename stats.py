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

from __future__ import annotations

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
            if '-' in sub:
                # Inclusive range
                try:
                    start_str, end_str = sub.split('-', 1)
                    start, end = int(start_str), int(end_str)
                    if start > end:
                        start, end = end, start
                    years.extend(range(start, end + 1))
                except ValueError:
                    # Fall back to treating as single year if parsing fails
                    try:
                        years.append(int(sub))
                    except Exception:
                        raise ValueError(f"Invalid year/range specification: {sub}")
            else:
                try:
                    years.append(int(sub))
                except Exception:
                    raise ValueError(f"Invalid year specification: {sub}")
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
    df_30 = df.resample("30T").mean(numeric_only=True)
    # time interpolation, fill both directions to avoid NaNs
    df_30["Air Temp (F)"] = df_30["Air Temp (F)"].interpolate(method="time", limit_direction="both")
    df_30["date"] = df_30.index.date
    df_30["h_of_day"] = df_30.index.hour + df_30.index.minute / 60.0

    grid = np.arange(0, 24, 0.5)
    daily_obs: Dict[datetime.date, np.ndarray] = {}
    for d, g in df_30.groupby("date"):
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
    mask = df_wx["datetime_local"].dt.year.isin(years) & (df_wx["datetime_local"].dt.month == month)
    df_mon = df_wx.loc[mask, ["datetime_local", "Air Temp (F)"]].copy()
    if df_mon.empty:
        raise ValueError(
            "No data found for the specified month/year(s) after timezone conversion. "
            f"month={month}, years={years}, tz={tz_name}"
        )

    # Build 30‑min daily vectors
    grid, daily_obs = resample_to_half_hour(df_mon)

    # Boundary on same grid
    bnd_vec = np.interp(grid, df_bnd["hour"].values, df_bnd["temp"].values)
    bnd_peak_idx = int(np.nanargmax(bnd_vec))
    bnd_peak_temp = float(np.nanmax(bnd_vec))

    dates = sorted(daily_obs.keys())
    n_days = len(dates)

    # ---- Risk 1: daily peak vs boundary peak (no shift)
    k1 = 0
    for d in dates:
        obs_peak = float(np.nanmax(daily_obs[d]))
        if obs_peak > bnd_peak_temp:
            k1 += 1
    (ci1_low, ci1_high), p1 = wilson_ci(k1, n_days)

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

        if float(np.nanmax(obs_vec)) > bnd_peak_temp:
            r1_exceed_dates.append(d)

    (ci2_low, ci2_high), p2 = wilson_ci(k2, n_days)

    # ---- File name helpers
    # Determine a label for years in output names
    sorted_years = sorted(years)
    if len(sorted_years) == 1:
        years_label = str(sorted_years[0])
    else:
        contiguous = all(
            sorted_years[i] + 1 == sorted_years[i + 1] for i in range(len(sorted_years) - 1)
        )
        if contiguous:
            years_label = f"{sorted_years[0]}-{sorted_years[-1]}"
        else:
            years_label = "_".join(str(y) for y in sorted_years)

    stem = f"{station}_{years_label}{month:02d}"

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
    for d in dates:
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

    # Severity histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degree_hours, bins=12, edgecolor="black")
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
                    f"• <b>Risk 1 (Peak exceedance)</b>: {k1} of {n_days} days exceeded the boundary peak. "
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
                    f"• <b>Risk 2 (Thermal loading, {risk2_window_hours:.0f} h)</b>: {k2} of {n_days} days contained a continuous {risk2_window_hours:.0f}-hour exceedance after peak alignment. "
                    f"The estimated probability that a randomly selected day from the selected period ({period_desc}) exceeds the thermal loading criterion is "
                    f"<b>{p2 * 100:.1f}%</b>. "
                    f"With 95% confidence, the true probability lies between <b>{ci2_low * 100:.1f}%</b> and <b>{ci2_high * 100:.1f}%</b>."
                ),
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 12))
        story.append(Image(plot_severity_hist, width=400, height=250))
        story.append(Spacer(1, 12))

        doc.build(story)

    return CaseStudyOutputs(
        pdf=pdf_path,
        csv_risk1=csv_risk1,
        csv_risk2=csv_risk2,
        plot_risk1_examples=plot_risk1_examples,
        plot_risk1_stacked=plot_risk1_stacked,
        plot_risk2_examples=plot_risk2_examples,
        plot_severity_hist=plot_severity_hist,
        stats={
            "n_days": n_days,
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
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate temperature vs boundary case studies.")
    parser.add_argument("--weather", required=True, help="Path to weather CSV")
    parser.add_argument("--boundary", required=True, help="Path to boundary CSV (hour,temp)")
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
        print("Severity histogram:", outputs.plot_severity_hist)
        print("Stats:", outputs.stats)


if __name__ == "__main__":
    main()