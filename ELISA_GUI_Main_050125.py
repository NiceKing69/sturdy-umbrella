import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import openpyxl
import matplotlib.pyplot as plt
from tksheet import Sheet


class ELISAGUI:
    def __init__(self, master):
        """
        Initialize the main GUI window with controls for
        selecting data, choosing analysis methods, and displaying results.
        """
        self.master = master
        master.title("ELISA Analysis Tool")
        master.geometry("1200x800")  # Set default window size

        # --- Internal state variables ---
        self.data = None
        self.standard_conc_cells = []    # coords of standard concentration cells
        self.standard_signal_cells = []  # coords of standard signal cells
        self.sample_name_cells = []      # coords of sample name cells
        self.sample_signal_cells = []    # coords of sample signal cells
        self.selecting_mode = None
        self.assay_type = tk.StringVar(value="normal")       # Normal or Competitive ELISA
        self.sample_presence = tk.StringVar(value="present") # Present or Not Present samples
        self.analysis_method = tk.StringVar(value="4PL")     # 4PL, 5PL, Linear, Log-Linear

        # --- Assay type selector ---
        frame_type = tk.Frame(master)
        frame_type.pack(pady=5)
        tk.Label(frame_type, text="Assay Type:").pack(side=tk.LEFT)
        tk.Radiobutton(frame_type, text="Normal ELISA", variable=self.assay_type,
                       value="normal").pack(side=tk.LEFT)
        tk.Radiobutton(frame_type, text="Competitive ELISA", variable=self.assay_type,
                       value="competitive").pack(side=tk.LEFT)

        # --- Sample presence selector ---
        frame_pres = tk.Frame(master)
        frame_pres.pack(pady=5)
        tk.Label(frame_pres, text="Sample Status:").pack(side=tk.LEFT)
        tk.Radiobutton(frame_pres, text="Present", variable=self.sample_presence,
                       value="present").pack(side=tk.LEFT)
        tk.Radiobutton(frame_pres, text="Not Present", variable=self.sample_presence,
                       value="not_present").pack(side=tk.LEFT)

        # --- Analysis method selector ---
        frame_method = tk.Frame(master)
        frame_method.pack(pady=5)
        tk.Label(frame_method, text="Analysis Method:").pack(side=tk.LEFT)
        for method in ["4PL", "5PL", "Linear", "Log-Linear"]:
            tk.Radiobutton(frame_method, text=method,
                           variable=self.analysis_method, value=method).pack(side=tk.LEFT)

        # --- Control buttons ---
        self.load_btn = tk.Button(
            master, text="Start with Loading Excel File",
            command=self.load_file)
        self.load_btn.pack(pady=5)

        self.choose_standard_conc_btn = tk.Button(
            master, text="Choose Standard Concentration",
            command=self.set_standard_conc_mode, state=tk.DISABLED)
        self.choose_standard_conc_btn.pack(pady=2)

        self.choose_standard_btn = tk.Button(
            master, text="Choose Standard Signal",
            command=self.set_standard_signal_mode, state=tk.DISABLED)
        self.choose_standard_btn.pack(pady=2)

        self.choose_sample_name_btn = tk.Button(
            master, text="Choose Sample Name",
            command=self.set_sample_name_mode, state=tk.DISABLED)
        self.choose_sample_name_btn.pack(pady=2)

        self.choose_sample_btn = tk.Button(
            master, text="Choose Sample Signal",
            command=self.set_sample_signal_mode, state=tk.DISABLED)
        self.choose_sample_btn.pack(pady=2)

        self.fit_btn = tk.Button(
            master, text="Run Analysis & Export",
            command=self.fit_and_export, state=tk.DISABLED)
        self.fit_btn.pack(pady=5)

        # --- Prompt label for user instructions ---
        self.prompt_label = tk.Label(master, text="")
        self.prompt_label.pack(pady=5)

        # --- Data sheet display ---
        self.sheet_frame = tk.Frame(master)
        self.sheet_frame.pack(fill=tk.BOTH, expand=True)
        self.sheet = Sheet(self.sheet_frame, width=1000, height=500)
        self.sheet.pack(fill=tk.BOTH, expand=True)
        self.sheet.enable_bindings("all")

    def load_file(self):
        """
        Prompt the user to select an Excel file, load it into a pandas DataFrame,
        and populate the sheet widget. Enable selection buttons afterward.
        """
        path = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx")])
        if not path:
            return

        # Read in the Excel data
        self.data = pd.read_excel(path, header=None)

        # Display in tksheet
        self.sheet.set_sheet_data(
            self.data.fillna("").astype(str).values.tolist()
        )
        self.sheet.redraw()

        # Enable controls
        for btn in (
            self.choose_standard_conc_btn,
            self.choose_standard_btn,
            self.choose_sample_name_btn,
            self.choose_sample_btn,
            self.fit_btn
        ):
            btn.config(state=tk.NORMAL)

    def set_standard_conc_mode(self):
        """
        Enter mode to select standard concentration cells.
        """
        self.selecting_mode = 'conc'
        self.prompt_label.config(
            text="Select one row or column for standard concentrations and press Enter"
        )
        self.sheet.bind("<Return>", self.confirm_selection)

    def set_standard_signal_mode(self):
        """
        Enter mode to select standard signal cells.
        Requires concentration cells first.
        """
        if not self.standard_conc_cells:
            messagebox.showwarning(
                "Need concentrations",
                "Select standard concentration cells first."
            )
            return
        self.selecting_mode = 'std_sig'
        self.prompt_label.config(
            text="Select one row or column for standard signals and press Enter"
        )
        self.sheet.bind("<Return>", self.confirm_selection)

    def set_sample_name_mode(self):
        """
        Enter mode to select sample name cells.
        """
        self.selecting_mode = 'samp_name'
        self.prompt_label.config(
            text="Select one row or column for sample names and press Enter"
        )
        self.sheet.bind("<Return>", self.confirm_selection)

    def set_sample_signal_mode(self):
        """
        Enter mode to select sample signal cells.
        Requires sample name cells first.
        """
        if not self.sample_name_cells:
            messagebox.showwarning(
                "Need names",
                "Select sample name cells first."
            )
            return
        self.selecting_mode = 'samp_sig'
        self.prompt_label.config(
            text="Select one row or column for sample signals and press Enter"
        )
        self.sheet.bind("<Return>", self.confirm_selection)

    def confirm_selection(self, event):
        """
        Handle Enter: verify row/column selection, highlight, and store coords.
        """
        sel = sorted(self.sheet.get_selected_cells())
        if not sel:
            messagebox.showwarning("No Selection","Select at least one cell.")
            return
        rows = {r for r, c in sel}
        cols = {c for r, c in sel}
        if not (len(rows) == 1 or len(cols) == 1):
            messagebox.showerror("Error","Select exactly one row or column.")
            return
        color_map = {'conc':'orange','std_sig':'lightblue','samp_name':'yellow','samp_sig':'lightgreen'}
        for r, c in sel:
            self.sheet.highlight_cells(row=r, column=c, bg=color_map[self.selecting_mode])
        if self.selecting_mode == 'conc':      self.standard_conc_cells = sel.copy()
        elif self.selecting_mode == 'std_sig': self.standard_signal_cells = sel.copy()
        elif self.selecting_mode == 'samp_name': self.sample_name_cells = sel.copy()
        elif self.selecting_mode == 'samp_sig':  self.sample_signal_cells = sel.copy()
        self.prompt_label.config(text='')
        self.selecting_mode = None
        self.sheet.unbind("<Return>")

    # --- Model functions ---
    def four_param_logistic(self, x, A, B, C, D):
        return D + (A - D)/(1 + (x/C)**B)

    def five_param_logistic(self, x, A, B, C, D, E):
        return D + (A - D)/(1 + (x/C)**B)**E

    def linear_model(self, x, m, b):
        return m*x + b

    def log_linear_model(self, x, m, b):
        return np.exp(b)*(x**m)

    def inverse_4pl(self, y, A, B, C, D):
        """
        Invert 4PL to estimate conc from signal.
        """
        try:
            ratio = (A - D)/(y - D) - 1
            if ratio <= 0: return np.nan
            return C*ratio**(1/B)
        except:
            return np.nan

    def fit_and_export(self):
        """
        Run the selected analysis, show tables, plot results, export to Excel.
        """
        try:
            if not self.standard_conc_cells or not self.standard_signal_cells:
                raise ValueError("Standards not set.")

            # Build standard map
            std_map = {}
            for r, c in self.standard_signal_cells:
                conc = None
                for rr, cc in self.standard_conc_cells:
                    if rr == r or cc == c:
                        conc = float(self.data.iat[rr, cc]); break
                if conc is None:
                    raise ValueError(f"No matching conc for signal at ({r},{c})")
                std_map.setdefault(conc, []).append(float(self.data.iat[r, c]))

            # Prepare data arrays
            concs = np.array(sorted(std_map))
            means = np.array([np.mean(std_map[c]) for c in concs])
            sds   = np.array([np.std(std_map[c])  for c in concs])
            all_c = np.concatenate([[c]*len(std_map[c]) for c in concs])
            all_s = np.concatenate([std_map[c] for c in concs])

            # Select and fit model
            method = self.analysis_method.get()
            if method == '4PL':
                popt, pcov = curve_fit(self.four_param_logistic, all_c, all_s, maxfev=10000)
                model_func = self.four_param_logistic
            elif method == '5PL':
                popt, pcov = curve_fit(self.five_param_logistic, all_c, all_s, maxfev=10000)
                model_func = self.five_param_logistic
            elif method == 'Linear':
                m, b = np.polyfit(all_c, all_s, 1)
                popt = np.array([m, b]); pcov = None
                model_func = self.linear_model
            else:  # Log-Linear
                mask = (all_c>0)&(all_s>0)
                m, b = np.polyfit(np.log(all_c[mask]), np.log(all_s[mask]), 1)
                popt = np.array([m, b]); pcov = None
                model_func = self.log_linear_model

            perr = (np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*len(popt))

            # Compute fit stats
            ypred = model_func(all_c, *popt)
            r2 = 1 - np.sum((all_s-ypred)**2)/np.sum((all_s-np.mean(all_s))**2)
            r  = np.sqrt(r2) if r2>=0 else np.nan
            rmse = np.sqrt(np.mean((all_s-ypred)**2))

            # Compute LoD signal
            if 0 in std_map:
                m0, s0 = np.mean(std_map[0]), np.std(std_map[0])
                lod_sig = m0 + 3*s0
            else:
                lod_sig = means[0]/10
            # Invert LoD to concentration with fallback logic
            if method == '4PL':
                lod_conc = self.inverse_4pl(lod_sig, *popt)
                if np.isnan(lod_conc) and 0 in std_map:
                    lod_sig_low = m0 - 3*s0
                    lod_conc = self.inverse_4pl(lod_sig_low, *popt)
                if np.isnan(lod_conc):
                    lod_conc = concs[-1]
            elif method == '5PL':
                lod_conc = concs[0] * 0.5
            elif method == 'Linear':
                lod_conc = (lod_sig - popt[1]) / popt[0]
            else:
                lod_conc = np.exp((np.log(lod_sig) - popt[1]) / popt[0])

            # Predict samples if present
            have_samples = bool(self.sample_name_cells and self.sample_signal_cells)
            if have_samples:
                names = [str(self.data.iat[r,c]) for r,c in self.sample_name_cells]
                sigs  = [float(self.data.iat[r,c]) for r,c in self.sample_signal_cells]
                if self.sample_presence.get() == 'present':
                    ests = []
                    for s in sigs:
                        if method in ['4PL','5PL']:
                            ests.append(self.inverse_4pl(s, *popt))
                        elif method == 'Linear':
                            ests.append((s - popt[1]) / popt[0])
                        else:
                            ests.append(np.exp((np.log(s) - popt[1]) / popt[0]))
                else:
                    ests = [np.nan] * len(sigs)
                # Show sample table
                df_s = pd.DataFrame({
                    "Sample Name": names,
                    "Signal": sigs,
                    "Estimated Concentration": ests
                })
                win1 = tk.Toplevel(self.master); win1.title("Sample Estimates")
                sh1 = Sheet(win1, width=500, height=200); sh1.pack(fill=tk.BOTH, expand=True)
                sh1.set_sheet_data([df_s.columns.tolist()] + df_s.values.tolist())

            # Prepare parameters table
            if method == '5PL': labels = ['A','B','C','D','E']
            elif method in ['Linear','Log-Linear']: labels = ['m','b']
            else: labels = ['A','B','C','D']
            params = [[labels[i], popt[i], perr[i] if i<len(perr) else ""] for i in range(len(popt))]
            params += [["R", r, ""], ["R²", r2, ""], ["RMSE", rmse, ""],
                       ["LoD Signal", lod_sig, ""], ["LoD Concentration", lod_conc, ""]]
            df_p = pd.DataFrame(params, columns=["Parameter","Estimate","Std Error"]);
            win2 = tk.Toplevel(self.master); win2.title("Parameters & LoD")
            sh2 = Sheet(win2, width=400, height=250); sh2.pack(fill=tk.BOTH, expand=True)
            sh2.set_sheet_data([df_p.columns.tolist()] + df_p.values.tolist())

            # Plot with larger figure and distinct sample color
            xf = np.linspace(concs.min()*0.5, concs.max()*1.5, 500)
            yf = model_func(xf, *popt)
            plt.figure(figsize=(12,8))
            plt.errorbar(concs, means, yerr=sds, fmt='o', label='Standards')
            if have_samples and self.sample_presence.get() == 'present':
                plt.scatter(sample_concs, sample_sigs, color='orange', label='Samples')
            plt.plot(xf, yf, label=method)
            plt.axhline(lod_sig, linestyle='--', label='LoD signal')
            plt.axvline(lod_conc, linestyle='--', label=f'LoD conc = {lod_conc:.2f}')
            plt.xlabel('Concentration'); plt.ylabel('Signal')
            plt.title(f"{self.assay_type.get().title()} ELISA ({method})")
            plt.text(0.05,0.95, f"R = {r:.4f}\nR² = {r2:.4f}", transform=plt.gca().transAxes, va='top')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=False)

            # Export to Excel
            save_path = filedialog.asksaveasfilename(defaultextension='*.xlsx', filetypes=[('Excel','*.xlsx')])
            if save_path:
                wb = openpyxl.Workbook(); ws = wb.active; ws.title = 'Results'
                if have_samples:
                    ws.append(['Sample Name','Signal','Estimated Concentration'])
                    for n,s,c in zip(names,sigs,ests): ws.append([n,s,c])
                    ws.append([])
                ws.append(['Parameter','Estimate','Std Error'])
                for row in params: ws.append(row)
                wb.save(save_path); messagebox.showinfo('Done', f'Saved to {save_path}')

        except RuntimeError:
            messagebox.showwarning(
                "Fit Warning",
                "Optimal parameters not found (maxfev reached). Consider choosing a different analysis method."
            )
        except Exception as e:
            messagebox.showerror('Error', str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = ELISAGUI(root)
    root.mainloop()