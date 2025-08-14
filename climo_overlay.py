import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Timezone handling
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    "from backports.zoneinfo import ZoneInfo  # fallback if needed (pip install backports.zoneinfo)
    
def build_august_climatology(
    raw_csv: str,
    years: int = 10,
    tz_name: str = "America/Chicago",
) -> pd.DataFrame:
    """
    Build August hourly climatology (local time) from climate.af.mil CSV.
    Expected columns in raw_csv: 'Date' (MM-DD-YYYY), 'Time (UTC)' (HH:MM), 'Air Temp (F)'.
    Returns a DataFrame with hour_local, mean, std, min, max, p05, p25, p75, p95.
    """
    df = pd.read_csv(raw_csv)
    # Combine to UTC datetime
    dt = pd.to_datetime(df["Date"] + " " + df["Time (UTC)"], format="%m-%d-%Y %H:%M", errors="coerce")
    df = df.assign(DATETIME=dt)

    # August only
    df = df[df["DATETIME"].dt.month == 8].copy()
    if df.empty:
        raise ValueError("No August rows found in the provided file.")

    # Last N years in the file
    latest_year = int(df["DATETIME"].dt.year.max())
    cutoff = latest_year - (years - 1)
    df = df[df["DATETIME"].dt.year >= cutoff].copy()

    # Temperature numeric
    df["Air Temp (F)"] = pd.to_numeric(df["Air Temp (F)"], errors="coerce")

    # Convert to local time & get local hour (0..23)
    df["DATETIME_LOCAL"] = df["DATETIME"].dt.tz_localize("UTC").dt.tz_convert(ZoneInfo(tz_name))
    df["hour_local"] = df["DATETIME_LOCAL"].dt.hour

    # Aggregate statistics by local hour
    def hour_agg(group: pd.DataFrame) -> pd.Series:
        vals = group["Air Temp (F)"].dropna().to_numpy()
        if vals.size == 0:
            return pd.Series(
                {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                 "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan}
            )
        return pd.Series({
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
        })

    hourly = df.groupby("hour_local", as_index=False).apply(hour_agg, include_groups=False)
    # In case hour_local is missing rows, reindex to 0..23
    hourly = hourly.set_index("hour_local").reindex(range(24)).reset_index()
    hourly.rename(columns={"index": "hour_local"}, inplace=True)

    hourly.attrs["latest_year"] = latest_year
    hourly.attrs["start_year"] = cutoff
    return hourly


def interpolate_to(test_hours: np.ndarray, hourly_climo: pd.DataFrame) -> pd.DataFrame:
    """Interpolate climatology columns to match the requested hours (e.g., 0, 0.5, …, 23.5).
    Ensures the returned frame has a column named 'hour'.
    """
    cols_to_interp = ["mean", "std", "min", "max", "p05", "p25", "p75", "p95"]
    climo = hourly_climo.set_index("hour_local")[cols_to_interp].copy()

    # Reindex to the target hours and interpolate linearly across hours
    climo = climo.reindex(test_hours)

    # Make sure the index has a name so reset_index produces a column named 'hour'
    climo.index = pd.Index(climo.index, name="hour")

    climo = climo.interpolate(method="linear", limit_direction="both").reset_index()
    return climo


def load_test_profile(csv_path: str) -> pd.DataFrame:
    """
    Expect a CSV with columns:
      hour  -> 0, 0.5, 1.0, ..., 23.5  (local hour)
      temp  -> temperature in F
    """
    test = pd.read_csv(csv_path)
    test.columns = [c.strip().lower() for c in test.columns]
    if "hour" not in test or "temp" not in test:
        raise ValueError("Test profile must have columns: hour, temp")

    test["hour"] = pd.to_numeric(test["hour"], errors="coerce")
    test["temp"] = pd.to_numeric(test["temp"], errors="coerce")
    test = test.sort_values("hour").dropna(subset=["hour", "temp"]).reset_index(drop=True)
    return test


def overlay_and_metrics(test: pd.DataFrame, climo_interp: pd.DataFrame, outdir: str,
                        title_prefix: str = "Del Rio AFB (KDLF)"):
    merged = pd.merge(test, climo_interp, on="hour", how="left")

    x = merged["hour"].to_numpy(float)
    y_test = merged["temp"].to_numpy(float)
    y_mean = merged["mean"].to_numpy(float)
    y_std  = merged["std"].to_numpy(float)
    p05, p25, p75, p95 = [merged[c].to_numpy(float) for c in ["p05", "p25", "p75", "p95"]]

    # --- Overlay chart ---
    plt.figure(figsize=(13, 6.5))
    plt.fill_between(x, p05, p95, alpha=0.12, label="5–95%")
    plt.fill_between(x, p25, p75, alpha=0.25, label="25–75% (IQR)")
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.20, label="±1 Std Dev")
    plt.plot(x, y_mean, label="Mean Temp (°F)")
    plt.plot(x, y_test, linewidth=2.5, marker="o", label="Chamber Profile")

    plt.title(f"Chamber Profile vs August Climatology — {title_prefix}")
    plt.xlabel("Hour of Day (Local Time)")
    plt.ylabel("Temperature (°F)")
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    overlay_path = os.path.join(outdir, "overlay_chamber_vs_climo.png")
    plt.savefig(overlay_path, dpi=160)
    plt.close()

    # --- Residuals & metrics ---
    resid = y_test - y_mean
    mae = float(np.nanmean(np.abs(resid)))
    rmse = float(np.sqrt(np.nanmean(resid ** 2)))
    amp_test = float(np.nanmax(y_test) - np.nanmin(y_test))
    amp_climo = float(np.nanmax(y_mean) - np.nanmin(y_mean))
    tmax_test = float(x[np.nanargmax(y_test)])
    tmax_climo = float(x[np.nanargmax(y_mean)])
    phase_shift = tmax_test - tmax_climo

    plt.figure(figsize=(13, 4))
    plt.bar(x, resid)
    plt.axhline(0, linewidth=1, color="black")
    plt.title("Hourly Residuals (Test - Climatology Mean)")
    plt.xlabel("Hour of Day (Local Time)")
    plt.ylabel("ΔTemp (°F)")
    plt.xticks(np.arange(0, 25, 2))
    plt.tight_layout()
    resid_path = os.path.join(outdir, "residuals.png")
    plt.savefig(resid_path, dpi=160)
    plt.close()

    # Save merged table too
    merged_path = os.path.join(outdir, "merged_test_vs_climo.csv")
    merged.to_csv(merged_path, index=False)

    # Print summary
    print("=== FIT METRICS ===")
    print(f"MAE: {mae:.2f} °F")
    print(f"RMSE: {rmse:.2f} °F")
    print(f"Amplitude (test vs climo): {amp_test:.1f} vs {amp_climo:.1f} °F")
    print(f"Time of max (test vs climo): {tmax_test:g} vs {tmax_climo:g} h (local)")
    print(f"Phase shift (test - climo): {phase_shift:+.1f} h")
    print()
    print("Saved:")
    print(f" - {overlay_path}")
    print(f" - {resid_path}")
    print(f" - {merged_path}")


def main():
    ap = argparse.ArgumentParser(description="Overlay chamber test profile vs August hourly climatology (last N years).")
    ap.add_argument("--raw", required=True, help="Path to climate.af.mil raw CSV (e.g., KDLF_ICAO_20150101_20250101.csv)")
    ap.add_argument("--test", required=True, help="Path to test profile CSV with columns: hour, temp")
    ap.add_argument("--outdir", default="./outputs", help="Where to write PNG/CSV outputs")
    ap.add_argument("--years", type=int, default=10, help="Number of most recent years to average (default 10)")
    ap.add_argument("--tz", default="America/Chicago", help="IANA timezone for local hour (default America/Chicago)")
    args = ap.parse_args()

    # Build climatology
    climo = build_august_climatology(args.raw, years=args.years, tz_name=args.tz)
    # Save the climatology in case you want to inspect/reuse
    os.makedirs(args.outdir, exist_ok=True)
    climo_csv = os.path.join(args.outdir, f"aug_climatology_local_{climo.attrs.get('start_year','?')}_{climo.attrs.get('latest_year','?')}.csv")
    climo.to_csv(climo_csv, index=False)
    print(f"Saved climatology: {climo_csv}")

    # Load your chamber test profile
    test = load_test_profile(args.test)

    # Interpolate climatology to match the test hours (handles half-hours cleanly)
    climo_interp = interpolate_to(test["hour"].to_numpy(), climo)

    # Make overlay + residuals + metrics
    title_prefix = f"{args.tz} • {climo.attrs.get('start_year','?')}-{climo.attrs.get('latest_year','?')}"
    overlay_and_metrics(test, climo_interp, args.outdir, title_prefix=title_prefix)


if __name__ == "__main__":
    main()