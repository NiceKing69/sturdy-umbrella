import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import openpyxl
import matplotlib.pyplot as plt
from tksheet import Sheet

class ELISAGUI:
    def __init__(self, master):
        self.master = master
        master.title("ELISA 4PL Analysis Tool")

        self.data = None
        self.file_path = None
        self.standard_cells = []
        self.sample_cells = []
        self.selecting_mode = None

        self.load_btn = tk.Button(master, text="Load Excel File", command=self.load_file)
        self.load_btn.pack(pady=5)

        self.prompt_label = tk.Label(master, text="")
        self.prompt_label.pack(pady=5)

        self.sheet_frame = tk.Frame(master)
        self.sheet_frame.pack(fill=tk.BOTH, expand=True)

        self.sheet = Sheet(self.sheet_frame, width=800, height=300)
        self.sheet.pack(fill=tk.BOTH, expand=True)
        self.sheet.enable_bindings("all")

        self.choose_standard_btn = tk.Button(master, text="Choose Standard", command=self.set_standard_mode, state=tk.DISABLED)
        self.choose_standard_btn.pack(pady=2)

        self.choose_sample_btn = tk.Button(master, text="Choose Sample", command=self.set_sample_mode, state=tk.DISABLED)
        self.choose_sample_btn.pack(pady=2)

        self.fit_btn = tk.Button(master, text="Fit 4PL and Export", command=self.fit_and_export, state=tk.DISABLED)
        self.fit_btn.pack(pady=5)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not path:
            return
        self.file_path = path
        self.data = pd.read_excel(path, header=None)
        self.sheet.set_sheet_data(self.data.fillna("").astype(str).values.tolist())
        self.choose_standard_btn.config(state=tk.NORMAL)
        self.choose_sample_btn.config(state=tk.NORMAL)
        self.fit_btn.config(state=tk.NORMAL)

    def set_standard_mode(self):
        self.selecting_mode = 'standard'
        self.prompt_label.config(text="Choose the standard cells and press Enter")
        self.sheet.bind("<Return>", self.confirm_selection)

    def set_sample_mode(self):
        self.selecting_mode = 'sample'
        self.prompt_label.config(text="Choose the sample cells and press Enter")
        self.sheet.bind("<Return>", self.confirm_selection)

    def confirm_selection(self, event):
        selected = self.sheet.get_selected_cells()
        cells = sorted((r, c) for r, c in selected)
        if not cells:
            messagebox.showwarning("No Selection", "Please select some cells.")
            return
        confirmed = messagebox.askyesno("Confirm", f"Confirm selected cells as {self.selecting_mode}?")
        if confirmed:
            for (r, c) in cells:
                self.sheet.highlight_cells(row=r, column=c, bg="lightblue" if self.selecting_mode == 'standard' else "lightgreen")
            if self.selecting_mode == 'standard':
                self.standard_cells = cells
            else:
                self.sample_cells = cells
            self.prompt_label.config(text="")
            self.sheet.unbind("<Return>")

    def four_param_logistic(self, x, A, B, C, D):
        return D + (A - D) / (1.0 + (x / C)**B)

    def inverse_4pl(self, y, A, B, C, D):
        try:
            ratio = (A - D) / (y - D) - 1
            if ratio <= 0:
                return np.nan
            return C * (ratio ** (1.0 / B))
        except (ZeroDivisionError, ValueError):
            return np.nan

    def fit_and_export(self):
        try:
            # Step 1: collect standard data
            std_map = {}
            for (r, c) in self.standard_cells:
                conc = simpledialog.askfloat("Input", f"Enter concentration for standard cell ({r}, {c}) with signal {self.data.iat[r, c]}")
                if conc is None:
                    raise ValueError("Concentration input cancelled.")
                std_map.setdefault(conc, []).append(float(self.data.iat[r, c]))

            concentrations = []
            mean_signals = []
            std_signals = []

            all_concs = []
            all_signals = []

            for conc, signals in sorted(std_map.items()):
                concentrations.append(conc)
                mean_signals.append(np.mean(signals))
                std_signals.append(np.std(signals))
                all_concs.extend([conc] * len(signals))
                all_signals.extend(signals)

            concentrations = np.array(concentrations)
            mean_signals = np.array(mean_signals)
            std_signals = np.array(std_signals)
            all_concs = np.array(all_concs)
            all_signals = np.array(all_signals)

            # Step 2: Fit 4PL
            popt, pcov = curve_fit(self.four_param_logistic, all_concs, all_signals, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))

            A, B, C, D = popt
            A_err, B_err, C_err, D_err = perr

            y_pred = self.four_param_logistic(all_concs, *popt)
            ss_res = np.sum((all_signals - y_pred) ** 2)
            ss_tot = np.sum((all_signals - np.mean(all_signals)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean((all_signals - y_pred) ** 2))

            # Step 3: predict samples
            sample_signals = [float(self.data.iat[r, c]) for (r, c) in self.sample_cells]
            predicted_concs = [self.inverse_4pl(y, *popt) for y in sample_signals]

            # Step 4: Plot
            x_fit = np.linspace(min(concentrations)*0.5, max(concentrations)*1.5, 500)
            y_fit_curve = self.four_param_logistic(x_fit, *popt)

            plt.figure()
            plt.errorbar(concentrations, mean_signals, yerr=std_signals, fmt='o', label="Standards (mean ± SD)", color="blue")
            plt.scatter(predicted_concs, sample_signals, label="Samples", color="green")
            plt.plot(x_fit, y_fit_curve, label="4PL Fit", color="red")
            plt.xlabel("Concentration")
            plt.ylabel("Signal")
            plt.title("4PL Curve Fit")
            plt.legend()
            plt.grid(True)
            param_text = f"A={A:.2f}±{A_err:.2f}\nB={B:.2f}±{B_err:.2f}\nC={C:.2f}±{C_err:.2f}\nD={D:.2f}±{D_err:.2f}\nR²={r_squared:.4f}\nRMSE={rmse:.4f}"
            plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
            plt.tight_layout()
            plt.show()

            # Step 5: Export to Excel
            save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if not save_path:
                return

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "4PL Results"
            ws.append(["Row", "Col", "Signal", "Predicted Concentration"])
            for (r, c), signal, conc in zip(self.sample_cells, sample_signals, predicted_concs):
                ws.append([r+1, c+1, signal, conc])

            ws.append([])
            ws.append(["Fitted Parameters", "Estimate", "Std Error"])
            ws.append(["A (Bottom)", A, A_err])
            ws.append(["B (Hill Slope)", B, B_err])
            ws.append(["C (EC50)", C, C_err])
            ws.append(["D (Top)", D, D_err])
            ws.append(["R²", r_squared, "N/A"])
            ws.append(["RMSE", rmse, "N/A"])

            wb.save(save_path)
            messagebox.showinfo("Success", f"4PL results saved to: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = ELISAGUI(root)
    root.mainloop()
