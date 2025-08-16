import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

import numpy as np
import pandas as pd

# Ensure repository root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import stats


def test_wilson_ci_basic():
    (ci_low, ci_high), p = stats.wilson_ci(5, 10)
    assert p == 0.5
    assert ci_low < p < ci_high
    assert np.isclose(ci_low, 0.2366, atol=1e-3)
    assert np.isclose(ci_high, 0.7634, atol=1e-3)


def test_parse_years_input():
    years = stats.parse_years_input(["2015-2017", "2019,2021", "2022"])
    assert years == [2015, 2016, 2017, 2019, 2021, 2022]


def test_parse_utc_local():
    dt = stats.parse_utc_local("01-02-2020", "12:00", "America/Chicago")
    assert dt.tzinfo is not None
    assert dt.hour == 6  # CST is UTC-6 in January


def test_qc_filter_days(tmp_path):
    rows = []
    base = datetime(2020, 1, 1)
    for h in range(24):
        rows.append({"datetime_local": base + timedelta(hours=h), "Air Temp (F)": float(h)})
    base2 = datetime(2020, 1, 2)
    for h in range(24):
        rows.append({"datetime_local": base2 + timedelta(hours=h), "Air Temp (F)": 0.0})
    df = pd.DataFrame(rows)
    filtered, keep_days, reasons = stats.qc_filter_days(df, 2.0, 2, 0.8, 24)
    assert len(keep_days) == 1
    assert base.date() in keep_days
    assert base2.date() not in keep_days
    assert base2.date() in reasons


def test_resample_to_half_hour():
    base = datetime(2020, 1, 1)
    df = pd.DataFrame({
        "datetime_local": [base, base + timedelta(hours=1), base + timedelta(hours=2)],
        "Air Temp (F)": [0.0, 10.0, 20.0],
    })
    grid, daily = stats.resample_to_half_hour(df)
    assert len(grid) == 48
    assert base.date() in daily
    vec = daily[base.date()]
    assert len(vec) == 48
    assert vec[0] == 0.0
    assert vec[-1] == 20.0


def test_plot_risk2_area_examples(tmp_path):
    idx = pd.date_range("2020-01-01", periods=48, freq="H")
    temps = [50.0] * 24 + [40.0] * 24
    df = pd.DataFrame({"temp": temps}, index=idx)
    boundary = np.array([45.0] * 24)
    outfile = stats.plot_risk2_area_examples(df, boundary, "TEST", 1, tmp_path, theta_area=10.0)
    assert Path(outfile).exists()


def test_generate_case_study(tmp_path):
    # Weather data for two days
    rows = []
    base = datetime(2020, 1, 1)
    for day in range(2):
        day_base = base + timedelta(days=day)
        for h in range(24):
            temp = 60 + 10 * np.sin(2 * np.pi * h / 24) + (10 if day == 0 else -10)
            rows.append({"Date": day_base.strftime("%m-%d-%Y"),
                         "Time (UTC)": f"{h:02d}:00",
                         "Air Temp (F)": temp})
    weather = pd.DataFrame(rows)
    weather_csv = tmp_path / "weather.csv"
    weather.to_csv(weather_csv, index=False)
    # Boundary: constant 60°F at 0.5h increments
    hours = np.arange(0, 24, 0.5)
    boundary = pd.DataFrame({"hour": hours, "temp": [60.0] * len(hours)})
    boundary_csv = tmp_path / "boundary.csv"
    boundary.to_csv(boundary_csv, index=False)

    outputs = stats.generate_case_study(
        weather_file=str(weather_csv),
        boundary_file=str(boundary_csv),
        station="TEST",
        month=1,
        years=[2020],
        tz_name="UTC",
        risk2_window_hours=2.0,
        risk2_area_thresh=10.0,
        outdir=str(tmp_path),
    )
    assert Path(outputs.pdf).exists()
    assert Path(outputs.csv_risk1).exists()
    assert Path(outputs.csv_risk2).exists()
    assert Path(outputs.plot_risk1_stacked).exists()
    assert Path(outputs.plot_severity_hist).exists()
    assert outputs.stats["n_days"] == 2


def test_parse_months_input():
    months = stats.parse_months_input(["7", "10-2", "5,6"])
    assert months == [1, 2, 5, 6, 7, 10, 11, 12]
