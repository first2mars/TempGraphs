import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import asdict
from zoneinfo import available_timezones
from stats import generate_case_study, parse_years_input


class StatsFrame(ttk.Frame):
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
        self.risk2_hours_var = tk.DoubleVar(value=2.0)
        self.risk2_area_var = tk.DoubleVar(value=10.0)
        self.qc_min_range_var = tk.DoubleVar(value=2.0)
        self.qc_min_unique_var = tk.IntVar(value=8)
        self.qc_max_flat_var = tk.DoubleVar(value=0.80)
        self.qc_min_samples_var = tk.IntVar(value=24)
        self.outdir_var = tk.StringVar(value="./outputs")
        self.title_var = tk.StringVar()

        row = 0
        ttk.Label(self, text="Weather CSV").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.weather_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_weather).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Boundary CSV").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.boundary_var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(self, text="Browse", command=self._browse_boundary).grid(row=row, column=2)

        row += 1
        ttk.Label(self, text="Station").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.station_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Months").grid(row=row, column=0, sticky="ne")
        self.month_list = tk.Listbox(self, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        for m in range(1, 13):
            self.month_list.insert(tk.END, m)
        self.month_list.grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Years (e.g., 2015-2020,2012)").grid(row=row, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.years_var).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(self, text="Timezone").grid(row=row, column=0, sticky="e")
        tz_values = sorted(available_timezones())
        self.tz_combo = ttk.Combobox(self, values=tz_values, textvariable=self.tz_var)
        self.tz_combo.grid(row=row, column=1, sticky="we")

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
        ttk.Button(self, text="Run", command=self._run).grid(row=row, column=0, columnspan=3, pady=5)

        row += 1
        ttk.Label(self, text="Log:").grid(row=row, column=0, sticky="nw")
        self.log = tk.Text(self, height=10, state="disabled")
        self.log.grid(row=row, column=1, columnspan=2, sticky="nsew")

        for i in range(3):
            self.columnconfigure(i, weight=1)
        self.rowconfigure(row, weight=1)

    def _browse_weather(self) -> None:
        path = filedialog.askopenfilename(title="Select weather CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if path:
            self.weather_var.set(path)

    def _browse_boundary(self) -> None:
        path = filedialog.askopenfilename(title="Select boundary CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if path:
            self.boundary_var.set(path)

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
        try:
            weather = self.weather_var.get().strip()
            boundary = self.boundary_var.get().strip()
            station = self.station_var.get().strip()
            if not weather or not boundary or not station:
                raise ValueError("Please provide weather, boundary, and station.")

            months = [int(self.month_list.get(i)) for i in self.month_list.curselection()]
            if not months:
                raise ValueError("Please select at least one month.")

            years_input = self.years_var.get().strip()
            years = parse_years_input([years_input]) if years_input else []
            if not years:
                raise ValueError("Please provide years.")

            for month in months:
                self._log(f"Running month {month}...")
                out = generate_case_study(
                    weather_file=weather,
                    boundary_file=boundary,
                    station=station,
                    month=month,
                    years=years,
                    tz_name=self.tz_var.get(),
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


def main() -> None:
    root = tk.Tk()
    root.title("TempGraphs")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    stats_frame = StatsFrame(notebook)
    notebook.add(stats_frame, text="Stats")

    root.mainloop()


if __name__ == "__main__":
    main()
