from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Use stats.py directly (faster & simpler than shelling out)
try:
    from stats import generate_case_study  # type: ignore
except Exception as e:  # pragma: no cover
    print("[ERROR] Could not import generate_case_study from stats.py. Is this script in the repo root?", file=sys.stderr)
    raise


@dataclass
class Target:
    metric_area_p: float
    metric_area_ci: Tuple[float, float]
    metric_peak_ci: Tuple[float, float]
    metric_win_ci: Tuple[float, float]
    direction: str
    window_h: float
    area_thresh: float


def load_target(summary_json_path: str) -> Target:
    with open(summary_json_path, "r", encoding="utf-8") as fh:
        j = json.load(fh)
    return Target(
        metric_area_p=float(j["risk2_area"]["p"]),
        metric_area_ci=(float(j["risk2_area"]["ci"][0]), float(j["risk2_area"]["ci"][1])),
        metric_peak_ci=(float(j["risk1"]["ci"][0]), float(j["risk1"]["ci"][1])),
        metric_win_ci=(float(j["risk2_window"]["ci"][0]), float(j["risk2_window"]["ci"][1])),
        direction=str(j.get("direction", "above")),
        window_h=float(j["risk2_window"]["hours"]),
        area_thresh=float(j["risk2_area"]["threshold_Fh"]),
    )


def parse_range(spec: str) -> List[float]:
    """Parse 'start:end:step' (or 'start:end'). Inclusive if exactly on a step."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"Bad range spec: {spec} (expected start:end[:step])")
    start = float(parts[0]); end = float(parts[1])
    step = float(parts[2]) if len(parts) == 3 else 1.0
    if step == 0:
        return [start]
    n = int(math.floor((end - start) / step + 1e-9)) + 1
    vals = [start + i * step for i in range(max(n, 1))]
    return [round(v, 10) for v in vals if (step > 0 and v <= end + 1e-9) or (step < 0 and v >= end - 1e-9)]


@dataclass
class BaseSpec:
    station: str
    weather: str
    tz: str
    years: List[int]


def read_bases_csv(path: str, years_default: List[int]) -> List[BaseSpec]:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    for r in ("station", "weather", "tz"):
        if r not in cols:
            raise ValueError(f"{path} missing required column: {r}")
    out: List[BaseSpec] = []
    for _, r in df.iterrows():
        station = str(r[cols["station"]]).strip()
        weather = str(r[cols["weather"]]).strip()
        tz = str(r[cols["tz"]]).strip()
        years_str = str(r[cols.get("years", None)]) if ("years" in cols) else ""
        yrs: List[int] = []
        if years_str and years_str.lower() != "nan":
            for tok in str(years_str).split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok:
                    a, b = tok.split("-", 1)
                    a, b = int(a), int(b)
                    if a > b:
                        a, b = b, a
                    yrs.extend(list(range(a, b + 1)))
                else:
                    yrs.append(int(tok))
        else:
            yrs = years_default
        out.append(BaseSpec(station=station, weather=weather, tz=tz, years=sorted(set(yrs))))
    return out


def transform_profile(seed_csv: str, a: float, b: float, out_csv: str) -> None:
    """(α,β) transform about the mean:
       T'(h) = mean(T) + α * (T(h) − mean(T)) + β
    """
    df = pd.read_csv(seed_csv)
    cols = {c.strip().lower(): c for c in df.columns}
    if not {"hour", "temp"}.issubset(cols.keys()):
        raise ValueError("Boundary CSV must have columns: hour,temp")
    hour_col, temp_col = cols["hour"], cols["temp"]
    h = pd.to_numeric(df[hour_col], errors="coerce").to_numpy()
    t = pd.to_numeric(df[temp_col], errors="coerce").to_numpy()
    mask = np.isfinite(h) & np.isfinite(t)
    h = h[mask]; t = t[mask]
    mu = float(np.mean(t))
    t_prime = mu + a * (t - mu) + b
    out = pd.DataFrame({"hour": h, "temp": t_prime})
    out.to_csv(out_csv, index=False)


def objective_for_base(
    stats: Dict, target: Target, w_peak: float, w_win: float, w_area: float, *, direction: str
) -> float:
    """Objective per base.

    For BELOW (cold) scenarios we treat the target CIs as *upper bounds* (don’t exceed).
    We do *not* reward being far below the target; a separate global beta-regularizer
    will push the profile up subject to these hinges. For ABOVE, keep the symmetric
    least-squares match to target probabilities.
    """
    J = 0.0

    # --- Risk 2 (area)
    p_area = float(stats["risk2_area"]["p"]) if stats.get("risk2_area") else float("nan")
    lo_a, hi_a = target.metric_area_ci
    if direction == "below":
        # Hinge only if we exceed the upper CI; otherwise no penalty (feasible region)
        if p_area > hi_a:
            J += w_area * (p_area - hi_a) ** 2
    else:
        # ABOVE: symmetric match to target p
        J += w_area * (p_area - target.metric_area_p) ** 2

    # --- Risk 1 (peak/trough)
    if w_peak > 0 and stats.get("risk1"):
        p_peak = float(stats["risk1"]["p"])
        lo_p, hi_p = target.metric_peak_ci
        if direction == "below":
            # Hinge only if exceeding the upper CI
            if p_peak > hi_p:
                J += w_peak * (p_peak - hi_p) ** 2
        else:
            # ABOVE: symmetric match
            if p_peak < lo_p:
                J += w_peak * (lo_p - p_peak) ** 2
            elif p_peak > hi_p:
                J += w_peak * (p_peak - hi_p) ** 2

    # --- Risk 2 (window)
    if w_win > 0 and stats.get("risk2"):
        p_win = float(stats["risk2"]["p"])
        lo_w, hi_w = target.metric_win_ci
        if direction == "below":
            if p_win > hi_w:
                J += w_win * (p_win - hi_w) ** 2
        else:
            if p_win < lo_w:
                J += w_win * (lo_w - p_win) ** 2
            elif p_win > hi_w:
                J += w_win * (p_win - hi_w) ** 2

    return float(J)


def run_grid(
    *,
    month: int,
    years: List[int],
    bases: List[BaseSpec],
    seed_boundary: str,
    target: Target,
    risk_direction: str,
    grid_a: List[float],
    grid_b: List[float],
    w_peak: float,
    w_win: float,
    w_area: float,
    risk2_hours: float,
    area_thresh: float,
    outdir: str,
    risk2_peak_window: Optional[Tuple[int, int]] = None,
    risk2_window_in_peak_only: bool = False,
    risk2_min_peak_delta: float = 0.0,
) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    os.makedirs(outdir, exist_ok=True)

    rows = []
    best_score = float("inf")
    best_ab = (None, None)

    with tempfile.TemporaryDirectory() as td:
        total = len(grid_a) * len(grid_b)
        done = 0
        for a in grid_a:
            for b in grid_b:
                total_J = 0.0
                # Transform once per (α,β)
                prof_path = os.path.join(td, f"profile_a{a:.3f}_b{b:.3f}.csv")
                transform_profile(seed_boundary, a, b, prof_path)

                # Evaluate candidate across all bases
                for base in bases:
                    outputs = generate_case_study(
                        weather_file=base.weather,
                        boundary_file=prof_path,
                        station=base.station,
                        month=month,
                        years=base.years,
                        tz_name=base.tz,
                        risk2_window_hours=risk2_hours,
                        risk2_area_thresh=area_thresh,
                        risk_direction=risk_direction,
                        risk2_peak_window=risk2_peak_window,
                        risk2_window_in_peak_only=risk2_window_in_peak_only,
                        risk2_min_peak_delta=risk2_min_peak_delta,
                        outdir=os.path.join(outdir, f"{base.station}"),
                        report_title=None,
                    )
                    total_J += objective_for_base(
                        outputs.stats, target, w_peak, w_win, w_area, direction=risk_direction
                    )

                rows.append({"alpha": a, "beta": b, "objective": total_J})
                # For BELOW scenarios, prefer warmer (higher beta) profiles as long as constraints are met.
                # Use a very small regularizer so it only breaks ties within the feasible region.
                if risk_direction == "below":
                    total_J += 1e-3 * (-b)
                if total_J < best_score:
                    best_score = total_J
                    best_ab = (a, b)
                done += 1
                print(f"[OPT] Evaluated (a={a:.3f}, b={b:.3f})  [{done}/{total}]  best_objective={best_score:.6f}")

    df = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    if best_ab == (None, None):
        raise RuntimeError("No candidates evaluated; check inputs.")
    return df, (float(best_ab[0]), float(best_ab[1]))


def main():
    ap = argparse.ArgumentParser(
        description="Optimize a reduced test profile (alpha, beta) to match an EDW target across multiple bases."
    )
    ap.add_argument("--target-summary", required=True, help="EDW target summary.json produced by stats.py")
    ap.add_argument("--seed-boundary", required=True, help="Seed boundary CSV (hour,temp) to scale/offset")
    ap.add_argument("--bases-csv", required=True, help="CSV: station,weather,tz[,years]")
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--years", type=str, required=True, help="e.g., 2015-2025 or 2015,2017,2019")
    ap.add_argument("--risk-direction", choices=["above", "below"], default="above")
    ap.add_argument("--risk2-hours", type=float, default=2.0)
    ap.add_argument("--risk2-area-thresh", type=float, default=10.0)
    ap.add_argument("--grid-a", default="0.60:1.05:0.05", help="alpha range (default 0.60:1.05:0.05)")
    ap.add_argument("--grid-b", default="-6:6:1", help="beta °F range (default -6:6:1)")
    ap.add_argument("--grid-b-start", type=float, help="beta grid start (°F), use with --grid-b-end [and --grid-b-step]")
    ap.add_argument("--grid-b-end", type=float, help="beta grid end (°F), use with --grid-b-start [and --grid-b-step]")
    ap.add_argument("--grid-b-step", type=float, default=1.0, help="beta grid step (°F); default 1.0")
    ap.add_argument("--w-peak", type=float, default=0.0, help="weight for peak CI hinge penalty")
    ap.add_argument("--w-window", type=float, default=0.0, help="weight for window CI hinge penalty")
    ap.add_argument("--w-area", type=float, default=1.0, help="weight for area risk match")
    ap.add_argument("--outdir", default="./outputs/optimize")
    ap.add_argument("--risk2-peak-window", nargs=2, type=int, metavar=("START_HR","END_HR"),
                help="Local-hour window [START END] for focusing Risk 2 metrics (wrap-around allowed, e.g., 21 3)")
    ap.add_argument("--risk2-window-in-peak-only", action="store_true",
                help="Apply the continuous window test only within the peak window")
    ap.add_argument("--risk2-min-peak-delta", type=float, default=0.0,
                help="Minimum required peak exceedance (°F) at aligned daily extreme for Risk 2 counting")
    args = ap.parse_args()

    # years -> list[int]
    def parse_years_token(tok: str) -> List[int]:
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            return list(range(a, b + 1))
        return [int(x.strip()) for x in tok.split(",") if x.strip()]

    years = sorted(set(parse_years_token(args.years)))
    bases = read_bases_csv(args.bases_csv, years_default=years)
    if not bases:
        raise SystemExit("No bases parsed from --bases-csv")

    target = load_target(args.target_summary)

    # Soft consistency checks
    if args.risk_direction != target.direction:
        print(f"[WARN] --risk-direction ({args.risk_direction}) != target.direction ({target.direction}). Proceeding with CLI value.")
    if abs(args.risk2_hours - target.window_h) > 1e-6:
        print(f"[WARN] --risk2-hours ({args.risk2_hours}) != target.window_h ({target.window_h}). Proceeding with CLI value.")
    if abs(args.risk2_area_thresh - target.area_thresh) > 1e-6:
        print(f"[WARN] --risk2-area-thresh ({args.risk2_area_thresh}) != target.area_thresh ({target.area_thresh}). Proceeding with CLI value.")

    grid_a = parse_range(args.grid_a)

    def build_range_from_start_end(start: float, end: float, step: float) -> List[float]:
        if step == 0:
            return [start]
        n = int(math.floor((end - start) / step + 1e-9)) + 1
        vals = [start + i * step for i in range(max(n, 1))]
        # Include end if we land on it within numeric fuzz
        out = [round(v, 10) for v in vals if (step > 0 and v <= end + 1e-9) or (step < 0 and v >= end - 1e-9)]
        return out

    grid_b: List[float]
    if args.grid_b is not None:
        try:
            grid_b = parse_range(args.grid_b)
        except Exception as e:
            if args.grid_b_start is None or args.grid_b_end is None:
                raise SystemExit(f"Failed to parse --grid-b ('{args.grid_b}'): {e}. Alternatively, specify --grid-b-start, --grid-b-end, and optional --grid-b-step.")
            grid_b = build_range_from_start_end(args.grid_b_start, args.grid_b_end, args.grid_b_step)
    else:
        if args.grid_b_start is None or args.grid_b_end is None:
            raise SystemExit("Provide either --grid-b 'start:end[:step]' OR --grid-b-start and --grid-b-end [--grid-b-step].")
        grid_b = build_range_from_start_end(args.grid_b_start, args.grid_b_end, args.grid_b_step)

    print(f"[INFO] alpha grid: {grid_a}")
    print(f"[INFO] beta grid:  {grid_b}")

    leaderboard, (best_a, best_b) = run_grid(
        month=args.month,
        years=years,
        bases=bases,
        seed_boundary=args.seed_boundary,
        target=target,
        risk_direction=args.risk_direction,
        grid_a=grid_a,
        grid_b=grid_b,
        w_peak=args.w_peak,
        w_win=args.w_window,
        w_area=args.w_area,
        risk2_hours=args.risk2_hours,
        area_thresh=args.risk2_area_thresh,
        outdir=args.outdir,
        risk2_peak_window=tuple(args.risk2_peak_window) if args.risk2_peak_window else None,
        risk2_window_in_peak_only=args.risk2_window_in_peak_only,
        risk2_min_peak_delta=args.risk2_min_peak_delta,
    )

    os.makedirs(args.outdir, exist_ok=True)
    leaderboard_csv = os.path.join(args.outdir, f"leaderboard_m{args.month}.csv")
    leaderboard.to_csv(leaderboard_csv, index=False)

    # Emit the best reduced profile beside the leaderboard
    seed_name = os.path.splitext(os.path.basename(args.seed_boundary))[0]
    reduced_csv = os.path.join(args.outdir, f"{seed_name}_reduced_m{args.month}_a{best_a:.3f}_b{best_b:.3f}.csv")
    transform_profile(args.seed_boundary, best_a, best_b, reduced_csv)

    # Manifest
    manifest = {
        "target_summary": os.path.abspath(args.target_summary),
        "bases_csv": os.path.abspath(args.bases_csv),
        "seed_boundary": os.path.abspath(args.seed_boundary),
        "month": args.month,
        "years": years,
        "risk_direction": args.risk_direction,
        "risk2_hours": args.risk2_hours,
        "risk2_area_thresh": args.risk2_area_thresh,
        "grid_a": grid_a,
        "grid_b": grid_b,
        "weights": {"peak": args.w_peak, "window": args.w_window, "area": args.w_area},
        "best": {"alpha": best_a, "beta": best_b},
        "leaderboard_csv": leaderboard_csv,
        "reduced_profile_csv": reduced_csv,
        "peak_window": list(args.risk2_peak_window) if args.risk2_peak_window else None,
        "window_in_peak_only": bool(args.risk2_window_in_peak_only),
        "min_peak_delta_F": float(args.risk2_min_peak_delta),
    }
    with open(os.path.join(args.outdir, f"opt_manifest_m{args.month}.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("=== Optimization complete ===")
    print("Best (alpha, beta):", (best_a, best_b))
    print("Leaderboard:", leaderboard_csv)
    print("Reduced profile:", reduced_csv)


if __name__ == "__main__":
    main()
    
#python optimize_profile.py \
#  --target-summary ./outputs/targets/KEDW_2015-2024-07_summary.json \
#  --seed-boundary ./data/111FtestCorrected.csv \
#  --bases-csv ./data/bases_jul.csv \
#  --month 7 \
#  --years 2015-2024 \
#  --risk-direction above \
#  --risk2-hours 2 \
#  --risk2-area-thresh 10 \
#  --grid-a 0.60:1.05:0.05 \
#  --grid-b-start -6 --grid-b-end 6 --grid-b-step 1 \
#  --w-peak 0 \
#  --w-window 0 \
#  --w-area 1 \
#  --outdir ./outputs/optimize/jul"