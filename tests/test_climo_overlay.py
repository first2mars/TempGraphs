import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Ensure repository root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import climo_overlay as co

def make_sample_csv(tmp_path):
    rows = []
    for year in [2023, 2024]:
        for hour in range(24):
            date = f"08-01-{year}"
            time = f"{hour:02d}:00"
            rows.append({"Date": date, "Time (UTC)": time, "Air Temp (F)": float(hour)})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_build_monthly_climatology(tmp_path):
    csv_path = make_sample_csv(tmp_path)
    climo = co.build_monthly_climatology(str(csv_path), month=8, years=2, tz_name="UTC")
    assert len(climo) == 24
    assert list(climo["hour_local"]) == list(range(24))
    assert climo.attrs["start_year"] == 2023
    assert climo.attrs["latest_year"] == 2024
    # mean should match the hour value
    assert np.allclose(climo["mean"].to_numpy(), np.arange(24))


def test_overlay_and_metrics(tmp_path):
    test = pd.DataFrame({"hour": [0, 1, 2], "temp": [10.0, 20.0, 30.0]})
    climo_interp = pd.DataFrame({
        "hour": [0, 1, 2],
        "mean": [5.0, 15.0, 25.0],
        "std": [1.0, 1.0, 1.0],
        "min": [0.0, 10.0, 20.0],
        "max": [10.0, 20.0, 30.0],
        "p05": [1.0, 11.0, 21.0],
        "p25": [2.0, 12.0, 22.0],
        "p75": [8.0, 18.0, 28.0],
        "p95": [9.0, 19.0, 29.0],
    })
    outdir = tmp_path
    co.overlay_and_metrics(test, climo_interp, str(outdir), title_prefix="TEST")
    merged_csv = outdir / "merged_test_vs_climo.csv"
    assert merged_csv.exists()
    merged = pd.read_csv(merged_csv)
    assert "residual" in merged.columns
    assert np.allclose(merged["residual"].to_numpy(), [5, 5, 5])
    assert (outdir / "overlay.png").exists()
    assert (outdir / "residuals.png").exists()


def test_infer_identifier_from_path():
    assert co.infer_identifier_from_path("/tmp/KDLF_test.csv") == "KDLF"
    # When fewer than four leading letters are present, the function returns the
    # first four characters of the basename (which may include punctuation).
    assert co.infer_identifier_from_path("/tmp/abc.csv") == "ABC."


def test_plot_composite_mean_std(tmp_path):
    df = pd.DataFrame({
        "hour_local": [0, 1],
        "KDLF_mean": [10.0, 20.0],
        "KDLF_std": [1.0, 1.0],
        "KDLF_p25": [9.0, 19.0],
        "KDLF_p75": [11.0, 21.0],
        "KDLF_p05": [8.0, 18.0],
        "KDLF_p95": [12.0, 22.0],
        "KEND_mean": [15.0, 25.0],
        "KEND_std": [1.0, 1.0],
        "KEND_p25": [14.0, 24.0],
        "KEND_p75": [16.0, 26.0],
        "KEND_p05": [13.0, 23.0],
        "KEND_p95": [17.0, 27.0],
    })
    test_df = pd.DataFrame({"hour": [0, 1], "temp": [11.0, 21.0]})
    out_png = tmp_path / "comp.png"
    co.plot_composite_mean_std(df, str(out_png), test=test_df)
    assert out_png.exists()


def test_find_latest_station_csv(tmp_path):
    d = tmp_path
    (d / "KDLF_20200101.csv").write_text("a")
    (d / "KDLF_20210101.csv").write_text("b")
    latest = co.find_latest_station_csv(str(d), "KDLF")
    assert latest.endswith("KDLF_20210101.csv")
    assert co.find_latest_station_csv(str(d), "XXXX") is None


def test_compute_per_year_hourly_means(tmp_path):
    csv_path = make_sample_csv(tmp_path)
    df, sy, ey = co.compute_per_year_hourly_means(str(csv_path), month=8, tz_name="UTC", years=2)
    assert sy == 2023 and ey == 2024
    assert df.shape == (48, 3)  # 24 hours * 2 years
    # ensure each year appears
    assert set(df["year"]) == {2023, 2024}


def test_plot_per_year_means(tmp_path):
    csv_path = make_sample_csv(tmp_path)
    df, sy, ey = co.compute_per_year_hourly_means(str(csv_path), month=8, tz_name="UTC", years=2)
    out_png = co.plot_per_year_means(df, ident="TEST", month=8, sy=sy, ey=ey, test_csv=None, outdir=str(tmp_path))
    assert Path(out_png).exists()
