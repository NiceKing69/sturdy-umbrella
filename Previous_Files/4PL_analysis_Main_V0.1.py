import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def four_param_logistic(x, A, B, C, D):
    """
    4PL function:
    A: minimum asymptote
    B: Hill's slope
    C: inflection point (EC50)
    D: maximum asymptote
    """
    return D + (A - D) / (1.0 + (x / C)**B)


def fit_4pl(x_data, y_data, p0=None):
    """
    Fit the 4PL curve to data.
    Returns popt and pcov.
    """
    if p0 is None:
        # initial guesses: min(y), 1, median(x), max(y)
        p0 = [min(y_data), 1.0, np.median(x_data), max(y_data)]
    popt, pcov = curve_fit(four_param_logistic, x_data, y_data, p0=p0, maxfev=10000)
    return popt, pcov


def calculate_lod(blank_values, factor=3):
    """
    Limit of Detection: mean(blank) + factor * std(blank)
    """
    return np.mean(blank_values) + factor * np.std(blank_values)


def param_confidence_intervals(popt, pcov, alpha=0.05):
    """
    Calculate approximate confidence intervals for fitted parameters.
    Returns a list of (low, high) for each parameter.
    """
    from scipy.stats import t
    dof = max(0, len(popt) - len(popt))
    tval = t.ppf(1.0 - alpha/2.0, dof) if dof > 0 else 1.96
    intervals = []
    for i, p in enumerate(popt):
        sigma = np.sqrt(pcov[i, i])
        intervals.append((p - tval * sigma, p + tval * sigma))
    return intervals


class ElisaGUI:
    def __init__(self, master):
        self.master = master
        master.title("ELISA Data Analysis Tool")

        # Data holders
        self.standard_df = None
        self.sample_df = None

        # GUI elements
        self.load_std_btn = tk.Button(master, text="Load Standard Curve", command=self.load_standard)
        self.load_std_btn.pack(pady=5)

        self.fit_btn = tk.Button(master, text="Fit 4PL Curve", command=self.fit_curve, state=tk.DISABLED)
        self.fit_btn.pack(pady=5)

        self.lod_btn = tk.Button(master, text="Calculate LoD", command=self.show_lod, state=tk.DISABLED)
        self.lod_btn.pack(pady=5)

        self.plot_btn = tk.Button(master, text="Plot Results", command=self.plot_results, state=tk.DISABLED)
        self.plot_btn.pack(pady=5)

    def load_standard(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            self.standard_df = pd.read_csv(file_path)
            if 'Concentration' not in self.standard_df.columns or 'Signal' not in self.standard_df.columns:
                raise ValueError("CSV must contain 'Concentration' and 'Signal' columns.")
            messagebox.showinfo("Loaded", "Standard curve data loaded successfully.")
            self.fit_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def fit_curve(self):
        x = self.standard_df['Concentration'].values
        y = self.standard_df['Signal'].values
        try:
            self.popt, self.pcov = fit_4pl(x, y)
            self.ci = param_confidence_intervals(self.popt, self.pcov)
            messagebox.showinfo("Success", f"4PL fit parameters:\n{self.popt}")
            self.lod_btn.config(state=tk.NORMAL)
            self.plot_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))

    def show_lod(self):
        # Assume blanks are first few rows or specified separately
        blank_vals = self.standard_df[self.standard_df['Concentration'] == 0]['Signal'].values
        if len(blank_vals) == 0:
            messagebox.showwarning("No blanks", "No blank (0 conc) values found.")
            return
        lod_val = calculate_lod(blank_vals)
        messagebox.showinfo("LoD", f"Limit of Detection: {lod_val:.3f}")

    def plot_results(self):
        fig, ax = plt.subplots()
        # plot raw data
        ax.scatter(self.standard_df['Concentration'], self.standard_df['Signal'], label='Data')
        # plot fit
        x_fit = np.logspace(np.log10(self.standard_df['Concentration'].min() + 1e-6),
                             np.log10(self.standard_df['Concentration'].max()), 100)
        y_fit = four_param_logistic(x_fit, *self.popt)
        ax.plot(x_fit, y_fit, label='4PL Fit')
        ax.set_xscale('log')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Signal')
        ax.legend()

        # embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)


if __name__ == '__main__':
    root = tk.Tk()
    app = ElisaGUI(root)
    root.mainloop()
