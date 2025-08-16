"""Tkinter GUI with tabs for case-study stats and climatology overlays."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import asdict
from zoneinfo import available_timezones
import os
import glob
import calendar
import pandas as pd
import numpy as np
import json
from stats import generate_case_study, parse_years_input
from climo_overlay import (
    build_monthly_climatology,
    infer_identifier_from_path,
    overlay_and_metrics,
)


class StatsFrame(ttk.Frame):
    """Collect options and run the stats case-study workflow."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self._build_widgets()

    def _build_widgets(self) -> None:
        # File pickers
        self.weather_var = tk.StringVar()
        self.boundary_var = tk.StringVar()
        self.station_var = tk.StringVar()
        self.years_var = tk.StringVar()
        self.tz_var = tk.StringVar(value="America/Los_Angeles")
        self.risk_direction_var = tk.StringVar(value="above")  # "above" or "below"
        self.risk2_hours_var = tk.DoubleVar(value=2.0)
        self.risk2_area_var = tk.DoubleVar(value=10.0)
        self.qc_min_range_var = tk.DoubleVar(value=2.0)
        self.qc_min_unique_var = tk.IntVar(value=8)
        self.qc_max_flat_var = tk.DoubleVar(value=0.80)
        self.qc_min_samples_var = tk.IntVar(value=24)
        self.outdir_var = tk.StringVar(value="./outputs")
        self.title_var = tk.StringVar()
        self.auto_direction_var = tk.BooleanVar(value=True)
        self._last_dir = os.getcwd()

        # New: preferred data directory to scan for station CSVs
        self.data_dir_var = tk.StringVar(value="./data")

        row = 0
        # Data directory (source for station weather CSVs)
        ttk.Label(self, text="Data Dir").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.data_dir_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_data_dir).grid(row=row, column=2)

        row += 1
        # Weather CSV is primarily selected by Station ID; combobox shows matches from Data Dir
        ttk.Label(self, text="Weather CSV").grid(row=row, column=0, sticky="e")
        self.weather_combo = ttk.Combobox(self, textvariable=self.weather_var, values=[], state="readonly", width=37)
        self.weather_combo.grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_weather).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Boundary CSV").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.boundary_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_boundary).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Station").grid(row=row, column=0, sticky="e")
        e_station = ttk.Entry(self, textvariable=self.station_var)
        e_station.grid(row=row, column=1, sticky="we")
        # React to station changes to populate Weather CSV options
        self.station_var.trace_add("write", lambda *args: self._on_station_changed())
        self.data_dir_var.trace_add("write", lambda *args: self._on_station_changed())

        row += 1
        ttk.Label(self, text="Months").grid(row=row, column=0, sticky="ne")
        self.month_list = tk.Listbox(self, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        for m in range(1, 13):
            self.month_list.insert(tk.END, m)
        self.month_list.grid(row=row, column=1, sticky="we")
        btns = ttk.Frame(self)
        btns.grid(row=row, column=2, sticky="n")
        ttk.Button(btns, text="All", width=6, command=self._select_all_months).pack(pady=2)
        ttk.Button(btns, text="Clear", width=6, command=self._clear_months).pack(pady=2)

        row += 1
        ttk.Label(self, text="Years (e.g., 2015-2020,2012)").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.years_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Timezone").grid(row=row, column=0, sticky="e")
        tz_values = sorted(available_timezones())
        self.tz_combo = ttk.Combobox(self, values=tz_values, textvariable=self.tz_var)
        self.tz_combo.grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Checkbutton(self, text="Auto risk direction", variable=self.auto_direction_var, command=self._on_auto_dir_toggle).grid(row=row, column=0, columnspan=2, sticky="w")

        row += 1
        ttk.Label(self, text="Risk Direction").grid(row=row, column=0, sticky="e")
        self.risk_direction_combo = ttk.Combobox(
            self,
            values=["above", "below"],
            textvariable=self.risk_direction_var,
            state="disabled",  # starts disabled when auto on
        )
        self.risk_direction_combo.grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Risk2 Hours").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.risk2_hours_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Risk2 Area Thresh").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.risk2_area_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="QC Min Range F").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.qc_min_range_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="QC Min Unique").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.qc_min_unique_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="QC Max Flat Frac").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.qc_max_flat_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="QC Min Samples").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.qc_min_samples_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Output Dir").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.outdir_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_outdir).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Title (optional)").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.title_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Button(self, text="Run", command=self._run).grid(row=row, column=0, pady=5)
        ttk.Button(self, text="Save Config", command=self._save_config).grid(row=row, column=1, sticky="w")
        ttk.Button(self, text="Load Config", command=self._load_config).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(self, text="Log:").grid(row=row, column=0, sticky="nw")
        self.log = tk.Text(self, height=10, state="disabled")
        self.log.grid(row=row, column=1, columnspan=2, sticky="nsew")
        ttk.Button(self, text="Clear Log", command=self._clear_log).grid(row=row, column=2, sticky="ne")

        # Initial population of weather candidates if station/data_dir pre-filled
        self._on_station_changed()

        for i in range(3):
            self.columnconfigure(i, weight=1)
        self.rowconfigure(row, weight=1)

    def _browse_data_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self._last_dir, title="Select data directory")
        if path:
            self._last_dir = path
            self.data_dir_var.set(path)

    def _browse_weather(self) -> None:
        path = filedialog.askopenfilename(initialdir=self._last_dir, title="Select weather CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if path:
            self._last_dir = os.path.dirname(path)
            self.weather_var.set(path)
            # If the file lives inside the current data dir and matches station, include it in combobox
            stn = (self.station_var.get() or "").strip().upper()
            if stn and os.path.dirname(path) == os.path.abspath(self.data_dir_var.get()):
                vals = list(self.weather_combo.cget("values"))
                if path not in vals:
                    vals.append(path)
                    self.weather_combo.configure(values=sorted(vals))

    def _on_station_changed(self) -> None:
        """Uppercase station and rescan data dir for matching CSVs, selecting the first hit."""
        stn = self.station_var.get().strip().upper()
        if stn != self.station_var.get():
            # write back uppercase without recursive storm: after_idle to avoid re-entrant trace
            self.after(0, lambda: self.station_var.set(stn))
        data_dir = self.data_dir_var.get().strip() or "./data"
        self._populate_weather_candidates(stn, data_dir)

    def _populate_weather_candidates(self, station: str, data_dir: str) -> None:
        try:
            candidates: list[str] = []
            if station and os.path.isdir(data_dir):
                # Patterns: {STN}_ICAO_*.csv and files containing _{STN}_
                pat1 = os.path.join(data_dir, f"{station}_ICAO_*.csv")
                pat2 = os.path.join(data_dir, f"*{station}*.csv")
                found = set(glob.glob(pat1)) | set(glob.glob(pat2))
                # Only files
                candidates = [p for p in sorted(found) if os.path.isfile(p)]
            # Update combobox
            self.weather_combo.configure(values=candidates)
            if candidates:
                # If current value not in list, set to first
                if self.weather_var.get() not in candidates:
                    self.weather_var.set(candidates[0])
            else:
                # leave current value as-is but ensure combobox shows empty list
                pass
        except Exception as e:
            self._log(f"[warn] Failed scanning data dir: {e}")

    def _browse_boundary(self) -> None:
        path = filedialog.askopenfilename(initialdir=self._last_dir, title="Select boundary CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if path:
            self._last_dir = os.path.dirname(path)
            self.boundary_var.set(path)
            self._auto_set_risk_direction_from_boundary(path)

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(initialdir=self._last_dir, title="Select output directory")
        if path:
            self._last_dir = os.path.dirname(path)
            self.outdir_var.set(path)

    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state="disabled")
        self.log.see(tk.END)

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")

    def _select_all_months(self) -> None:
        self.month_list.select_set(0, tk.END)

    def _clear_months(self) -> None:
        self.month_list.selection_clear(0, tk.END)

    def _on_auto_dir_toggle(self) -> None:
        if self.auto_direction_var.get():
            self.risk_direction_combo.configure(state="disabled")
            if self.boundary_var.get():
                self._auto_set_risk_direction_from_boundary(self.boundary_var.get())
        else:
            self.risk_direction_combo.configure(state="readonly")

    def _auto_set_risk_direction_from_boundary(self, path: str) -> None:
        if not self.auto_direction_var.get():
            return
        name = os.path.basename(path).lower()
        # Heuristics: filenames with neg, below, cold => below; pos, above, hot, heat => above
        below_keys = ("neg", "below", "cold", "winter")
        above_keys = ("pos", "above", "hot", "heat", "summer")
        direction = None
        if any(k in name for k in below_keys):
            direction = "below"
        elif any(k in name for k in above_keys):
            direction = "above"
        if direction:
            self.risk_direction_var.set(direction)
            self._log(f"[auto] Risk direction set to '{direction}' based on boundary: {os.path.basename(path)}")

    def _gather_config(self) -> dict:
        months = [int(self.month_list.get(i)) for i in self.month_list.curselection()]
        return {
            "weather": self.weather_var.get(),
            "boundary": self.boundary_var.get(),
            "station": self.station_var.get(),
            "months": months,
            "years": self.years_var.get(),
            "tz": self.tz_var.get(),
            "auto_direction": self.auto_direction_var.get(),
            "risk_direction": self.risk_direction_var.get(),
            "risk2_hours": self.risk2_hours_var.get(),
            "risk2_area": self.risk2_area_var.get(),
            "qc_min_range": self.qc_min_range_var.get(),
            "qc_min_unique": self.qc_min_unique_var.get(),
            "qc_max_flat": self.qc_max_flat_var.get(),
            "qc_min_samples": self.qc_min_samples_var.get(),
            "outdir": self.outdir_var.get(),
            "title": self.title_var.get(),
        }

    def _apply_config(self, cfg: dict) -> None:
        self.weather_var.set(cfg.get("weather", ""))
        self.boundary_var.set(cfg.get("boundary", ""))
        self.station_var.set(cfg.get("station", ""))
        self.years_var.set(cfg.get("years", ""))
        self.tz_var.set(cfg.get("tz", self.tz_var.get()))
        self.auto_direction_var.set(cfg.get("auto_direction", True))
        self.risk_direction_var.set(cfg.get("risk_direction", self.risk_direction_var.get()))
        self.risk2_hours_var.set(cfg.get("risk2_hours", self.risk2_hours_var.get()))
        self.risk2_area_var.set(cfg.get("risk2_area", self.risk2_area_var.get()))
        self.qc_min_range_var.set(cfg.get("qc_min_range", self.qc_min_range_var.get()))
        self.qc_min_unique_var.set(cfg.get("qc_min_unique", self.qc_min_unique_var.get()))
        self.qc_max_flat_var.set(cfg.get("qc_max_flat", self.qc_max_flat_var.get()))
        self.qc_min_samples_var.set(cfg.get("qc_min_samples", self.qc_min_samples_var.get()))
        self.outdir_var.set(cfg.get("outdir", self.outdir_var.get()))
        self.title_var.set(cfg.get("title", ""))
        # months selection
        self._clear_months()
        months = cfg.get("months", [])
        for m in months:
            try:
                idx = m - 1
                self.month_list.select_set(idx)
            except Exception:
                pass
        self._on_auto_dir_toggle()

    def _save_config(self) -> None:
        cfg = self._gather_config()
        path = filedialog.asksaveasfilename(defaultextension=".json", initialdir=self._last_dir, title="Save config", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self._last_dir = os.path.dirname(path)
            self._log(f"Saved config to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(initialdir=self._last_dir, title="Load config", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_config(cfg)
            self._last_dir = os.path.dirname(path)
            self._log(f"Loaded config from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run(self) -> None:
        """Execute stats workflow for each selected month."""

        try:
            weather = self.weather_var.get().strip()
            boundary = self.boundary_var.get().strip()
            station = self.station_var.get().strip().upper()
            # Refresh candidates in case user only typed station then hit Run
            self._populate_weather_candidates(station, self.data_dir_var.get().strip() or "./data")
            if not weather and self.weather_combo.cget("values"):
                # Auto-pick first candidate if empty
                first = self.weather_combo.cget("values")[0]
                self.weather_var.set(first)
                weather = first
            if not weather or not boundary or not station:
                raise ValueError("Please provide weather, boundary, and station.")

            months = [int(self.month_list.get(i)) for i in self.month_list.curselection()]
            if not months:
                raise ValueError("Please select at least one month.")

            years_input = self.years_var.get().strip()
            years = parse_years_input([years_input]) if years_input else []
            if not years:
                raise ValueError("Please provide years.")

            self._on_auto_dir_toggle()

            for month in months:
                self._log(f"Running month {month}...")
                out = generate_case_study(
                    weather_file=weather,
                    boundary_file=boundary,
                    station=station,
                    month=month,
                    years=years,
                    tz_name=self.tz_var.get(),
                    risk_direction=self.risk_direction_var.get(),
                    risk2_window_hours=self.risk2_hours_var.get(),
                    risk2_area_thresh=self.risk2_area_var.get(),
                    outdir=self.outdir_var.get() or None,
                    report_title=self.title_var.get() or None,
                    qc_min_range_f=self.qc_min_range_var.get(),
                    qc_min_unique=self.qc_min_unique_var.get(),
                    qc_max_flat_frac=self.qc_max_flat_var.get(),
                    qc_min_samples=self.qc_min_samples_var.get(),
                )
                for key, val in asdict(out).items():
                    self._log(f"{key}: {val}")
                self._log("Done")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self._log(f"Error: {exc}")


class ClimoOverlayFrame(ttk.Frame):
    """Run climatology builds and overlay plots for chamber tests."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self._build_widgets()

    def _build_widgets(self) -> None:
        self.raw_var = tk.StringVar()
        self.test_var = tk.StringVar()
        self.outdir_var = tk.StringVar(value="./outputs")
        self.years_var = tk.IntVar(value=10)
        self.start_year_var = tk.StringVar()
        self.end_year_var = tk.StringVar()
        self.month_var = tk.IntVar(value=8)
        self.tz_var = tk.StringVar(value="America/Chicago")
        self.avg_only_var = tk.BooleanVar(value=False)

        row = 0
        ttk.Label(self, text="Raw Station CSV").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.raw_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_raw).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Test CSV").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.test_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_test).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Output Dir").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.outdir_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_outdir).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Month").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.month_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Years").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.years_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Start Year").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.start_year_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="End Year").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.end_year_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Timezone").grid(row=row, column=0, sticky="e")
        tz_values = sorted(available_timezones())
        self.tz_combo = ttk.Combobox(self, values=tz_values, textvariable=self.tz_var)
        self.tz_combo.grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Checkbutton(self, text="Average Only", variable=self.avg_only_var).grid(row=row, column=0, columnspan=2, sticky="w")

        row += 1
        ttk.Label(self, text="Shaded region style").grid(row=row, column=0, sticky="e")
        self.shade_var = tk.StringVar(value="both")
        shade_frame = ttk.Frame(self)
        shade_frame.grid(row=row, column=1, sticky="w")
        for val, label in [("iqr", "IQR (25–75%)"), ("std", "±1 Std Dev"), ("both", "Both"), ("none", "None")]:
            ttk.Radiobutton(shade_frame, text=label, value=val, variable=self.shade_var).pack(side=tk.LEFT, padx=4)

        row += 1
        ttk.Button(self, text="Run", command=self._run).grid(row=row, column=0, columnspan=3, pady=5)

        row += 1
        ttk.Label(self, text="Log:").grid(row=row, column=0, sticky="nw")
        self.log = tk.Text(self, height=10, state="disabled")
        self.log.grid(row=row, column=1, columnspan=2, sticky="nsew")

        for i in range(3):
            self.columnconfigure(i, weight=1)
        self.rowconfigure(row, weight=1)

    def _browse_raw(self) -> None:
        path = filedialog.askopenfilename(title="Select station CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if path:
            self.raw_var.set(path)

    def _browse_test(self) -> None:
        path = filedialog.askopenfilename(title="Select test CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if path:
            self.test_var.set(path)

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.outdir_var.set(path)

    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state="disabled")
        self.log.see(tk.END)

    def _run(self) -> None:
        """Generate climatology and overlay plots for the chosen test profile."""

        try:
            raw = self.raw_var.get().strip()
            test_csv = self.test_var.get().strip()
            if not raw or not test_csv:
                raise ValueError("Please provide raw and test CSV files.")

            month = int(self.month_var.get())
            years = int(self.years_var.get())
            tz_name = self.tz_var.get().strip()
            sy = int(self.start_year_var.get()) if self.start_year_var.get() else None
            ey = int(self.end_year_var.get()) if self.end_year_var.get() else None

            climo = build_monthly_climatology(
                raw,
                month=month,
                years=years,
                tz_name=tz_name,
                start_year=sy,
                end_year=ey,
            )

            ident = infer_identifier_from_path(raw)
            mon_abbr = calendar.month_abbr[month].lower()
            station_outdir = os.path.join(self.outdir_var.get(), "stations", ident, mon_abbr)
            os.makedirs(station_outdir, exist_ok=True)
            climo_csv = os.path.join(station_outdir, "climo.csv")
            climo.to_csv(climo_csv, index=False)

            test = pd.read_csv(test_csv, dtype={"hour": "float64", "temp": "float64"}, low_memory=False)
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

            title_prefix = (
                f"{ident} — {calendar.month_name[month]} Climatology "
                f"• {climo.attrs.get('start_year','?')}-{climo.attrs.get('latest_year','?')}"
            )
            overlay_and_metrics(
                test,
                climo_interp,
                station_outdir,
                title_prefix=title_prefix,
                average_only=self.avg_only_var.get(),
                shaded=self.shade_var.get(),
                extreme_days=climo.attrs.get('extreme_days'),
                overlay_png_name=f"{ident}_{mon_abbr}_overlay.png",
                residuals_png_name=f"{ident}_{mon_abbr}_residuals.png",
            )
            self._log("Done")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self._log(f"Error: {exc}")



# --- CLIMO OVERLAY TAB (for multi-station overlay script) ---
def climo_overlay_tab(notebook):
    tab = tk.Frame(notebook)
    notebook.add(tab, text="Climo Overlay (Multi-Station)")

    # Data directory
    data_dir_label = tk.Label(tab, text="Data Directory (scan for station CSVs):")
    data_dir_label.pack(anchor="w")
    data_dir_var = tk.StringVar(value="./data")
    data_dir_entry = tk.Entry(tab, width=50, textvariable=data_dir_var)
    data_dir_entry.pack(anchor="w")
    def browse_data_dir():
        d = filedialog.askdirectory(title="Select data directory")
        if d:
            data_dir_var.set(d)
            scan_csvs()
            update_station_status()
    tk.Button(tab, text="Browse", command=browse_data_dir).pack(anchor="w")

    # Output directory
    tk.Label(tab, text="Output Directory:").pack(anchor="w")
    outdir_entry = tk.Entry(tab, width=50)
    outdir_entry.insert(0, "./outputs")
    outdir_entry.pack(anchor="w")
    def browse_outdir():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            outdir_entry.delete(0, tk.END)
            outdir_entry.insert(0, d)
    tk.Button(tab, text="Browse", command=browse_outdir).pack(anchor="w")

    # Timezone selector
    tk.Label(tab, text="Timezone:").pack(anchor="w")
    tz_var = tk.StringVar(value="America/Chicago")
    ttk.Combobox(tab, textvariable=tz_var, values=sorted(available_timezones())).pack(fill="x")

    # Station IDs input with CSV scan
    tk.Label(tab, text="Enter Station IDs (comma-separated):").pack(anchor="w")
    station_var = tk.StringVar()
    station_entry = tk.Entry(tab, width=50, textvariable=station_var)
    station_entry.pack(anchor="w")
    station_status = tk.Label(tab, text="", fg="gray")
    station_status.pack(anchor="w")

    # Detected CSVs listbox (populated dynamically)
    csv_label = tk.Label(tab, text="Detected CSV files:")
    csv_label.pack(anchor="w")
    csv_listbox = tk.Listbox(tab, selectmode=tk.MULTIPLE, height=6, exportselection=False)
    csv_listbox.pack(fill="x")

    def scan_csvs():
        csv_listbox.delete(0, tk.END)
        try:
            current_dir = data_dir_var.get()
            files = [f for f in os.listdir(current_dir) if f.lower().endswith('.csv')]
        except Exception:
            files = []
        for f in sorted(files):
            csv_listbox.insert(tk.END, f)
        csv_label.config(text=f"Detected CSV files in {data_dir_var.get() or './data'}:")
    scan_csvs()

    # Month
    tk.Label(tab, text="Month (1=Jan, 12=Dec):").pack(anchor="w")
    month_entry = tk.Entry(tab, width=10)
    month_entry.insert(0, "8")
    month_entry.pack(anchor="w")

    # Years
    tk.Label(tab, text="Years (number of most recent years):").pack(anchor="w")
    years_entry = tk.Entry(tab, width=10)
    years_entry.insert(0, "10")
    years_entry.pack(anchor="w")
    all_years_var = tk.BooleanVar(value=False)
    def toggle_years():
        years_entry.configure(state=(tk.DISABLED if all_years_var.get() else tk.NORMAL))
    tk.Checkbutton(tab, text="Use all available years", variable=all_years_var, command=toggle_years).pack(anchor="w")
    toggle_years()

    # Mode and options
    composite_var = tk.BooleanVar(value=True)
    avgonly_var = tk.BooleanVar(value=False)
    peryear_var = tk.BooleanVar(value=False)
    def on_mode_toggle():
        peryear_chk.configure(state=(tk.DISABLED if composite_var.get() else tk.NORMAL))
    tk.Checkbutton(tab, text="Composite (combine stations)", variable=composite_var, command=on_mode_toggle).pack(anchor="w")
    tk.Checkbutton(tab, text="Average only (hide bands)", variable=avgonly_var).pack(anchor="w")
    peryear_chk = tk.Checkbutton(tab, text="Per-year curves (single-station mode)", variable=peryear_var)
    peryear_chk.pack(anchor="w")
    on_mode_toggle()

    # Shaded region style
    tk.Label(tab, text="Shaded region style:").pack(anchor="w")
    shade_var = tk.StringVar(value="both")
    shade_row = tk.Frame(tab)
    shade_row.pack(anchor="w")
    for val, label in [("iqr", "IQR (25–75%)"), ("std", "±1 Std Dev"), ("both", "Both"), ("none", "None")]:
        tk.Radiobutton(shade_row, text=label, variable=shade_var, value=val).pack(side=tk.LEFT, padx=4)

    # Optional chamber test CSV
    tk.Label(tab, text="Chamber Test CSV (optional, hour,temp):").pack(anchor="w")
    test_entry = tk.Entry(tab, width=50)
    test_entry.pack(anchor="w")
    def browse_test_csv():
        f = filedialog.askopenfilename(title="Select test CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if f:
            test_entry.delete(0, tk.END)
            test_entry.insert(0, f)
    tk.Button(tab, text="Browse Test CSV", command=browse_test_csv).pack(anchor="w")

    # Helpers
    def station_id_exists(stn: str, directory: str) -> bool:
        stn = (stn or '').strip().upper()
        if not stn or not os.path.isdir(directory):
            return False
        try:
            for name in os.listdir(directory):
                if not name.lower().endswith('.csv'):
                    continue
                if name[:4].upper() == stn:
                    return True
        except Exception:
            return False
        return False

    def update_station_status(*_):
        ids = [s.strip().upper() for s in (station_var.get() or '').split(',') if s.strip()]
        found, missing = [], []
        d = data_dir_var.get()
        for s in ids:
            (found if station_id_exists(s, d) else missing).append(s)
        msg_parts = []
        if found:
            msg_parts.append("Found: " + ", ".join(found))
        if missing:
            msg_parts.append("Missing: " + ", ".join(missing))
        station_status.config(text=" | ".join(msg_parts) if msg_parts else '')

    station_var.trace_add('write', update_station_status)
    data_dir_var.trace_add('write', lambda *_: (scan_csvs(), update_station_status()))

    # Runner
    def run_climo_overlay():
        try:
            outdir = outdir_entry.get().strip()
            data_dir = data_dir_var.get().strip() or './data'
            if not outdir:
                messagebox.showerror('Error', 'Please provide an output directory.')
                return
            month = int(month_entry.get())
            years = (9999 if all_years_var.get() else int(years_entry.get()))
            station_ids = [s.strip().upper() for s in (station_var.get() or '').split(',') if s.strip()]
            if not station_ids and csv_listbox.curselection():
                selected = [csv_listbox.get(i) for i in csv_listbox.curselection()]
                station_ids = [os.path.splitext(f)[0].split('_')[0].upper() for f in selected]
            if not station_ids:
                messagebox.showerror('Error', 'Please enter at least one station ID or select CSV files.')
                return
            test_csv = test_entry.get().strip()
            cmd = [
                'python', 'climo_overlay.py',
                '--month', str(month),
                '--years', str(years),
                '--data_dir', data_dir,
                '--outdir', outdir,
                '--tz', tz_var.get(),
                '--shade', shade_var.get(),
            ]
            if composite_var.get():
                cmd += ['--composite', '--stations', ','.join(station_ids)]
                if test_csv:
                    cmd += ['--composite_test', test_csv]
            else:
                cmd += ['--station', ','.join(station_ids)]
                if test_csv:
                    cmd += ['--test', test_csv]
                if peryear_var.get():
                    cmd += ['--per_year']
            import subprocess
            subprocess.Popen(cmd)
            messagebox.showinfo('Started', f'Climo overlay script started.\nCheck {outdir} for results.')
        except Exception as exc:
            messagebox.showerror('Error', str(exc))

    tk.Button(tab, text="Run Climo Overlay", command=run_climo_overlay).pack(pady=6)
    update_station_status()

def main() -> None:
    root = tk.Tk()
    root.title("TempGraphs")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Stats tab
    stats_frame = StatsFrame(notebook)
    notebook.add(stats_frame, text="Stats")

    # Single-station climo overlay tab
    climo_frame = ClimoOverlayFrame(notebook)
    notebook.add(climo_frame, text="Climo Overlay")

    # Multi-station climo overlay tab (script wrapper)
    climo_overlay_tab(notebook)

    root.mainloop()


if __name__ == "__main__":
    try:
        print("[INFO] Launching TempGraphs GUI…")
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("Startup Error", str(exc))
        except Exception:
            pass