# ELISA Analysis Tool

A user-friendly Python GUI application for analyzing ELISA data and generating publication-quality standard curves and sample estimates.

## Features

* **Excel Import:** Load raw ELISA data directly from an `.xlsx` file (no headers required).
* **Interactive Selection:** Select standard concentrations, standard signals, sample names, and sample signals with just a few clicks.
* **Assay Types:** Supports both Normal and Competitive ELISA formats.
* **Multiple Fitting Methods:** 4-Parameter Logistic (4PL), 5-Parameter Logistic (5PL), Linear, and Log-Linear curve fitting.
* **Limit of Detection (LoD):** Automatic calculation of LoD based on blank or low-concentration standards.
* **Result Tables:** Pop-up tables displaying fit parameters, LoD, and (optional) sample concentration estimates.
* **Plotting:** Interactive Matplotlib plots of standards, fitted curves, LoD lines, and sample points.
* **Export:** Save results and parameters to a new Excel workbook (`.xlsx`).

---

## Requirements

* **Python 3.7+**
* **Dependencies:**

  * pandas
  * numpy
  * scipy
  * matplotlib
  * openpyxl
  * tksheet
  * tkinter (usually included with standard Python installs)

Install dependencies via pip:

```bash
pip install pandas numpy scipy matplotlib openpyxl tksheet
```

---

## Installation & Usage

1. **Clone or download** this repository onto your local machine.
2. **Install** the required Python packages (see Requirements above).
3. **Run** the GUI:

   ```bash
   python ELISA_GUI_Main.py
   ```

---

## Workflow Guide

1. **Load Data**
   Click **Start with Loading Excel File** and select your `.xlsx` file containing raw OD or signal values.

2. **Select Standards**

   * Click **Choose Standard Concentration**, select the row or column of known standard concentrations, and press Enter.
   * Click **Choose Standard Signal**, select the matching OD/signals for those standards, and press Enter.

3. **(Optional) Select Samples**

   * Click **Choose Sample Name** and select the row/column of sample identifiers, then press Enter.
   * Click **Choose Sample Signal** and select the corresponding signal values, then press Enter.

4. **Configure Analysis**

   * Choose **Assay Type**: Normal or Competitive ELISA.
   * Choose **Sample Status**: Present or Not Present (determines whether concentrations are estimated).
   * Select **Analysis Method**: 4PL, 5PL, Linear, or Log-Linear.

5. **Run & Export**
   Click **Run Analysis & Export**.

   * Pop-up tables will show fitting parameters, LoD, and (if samples present) estimated concentrations.
   * A plot window will display the standard curve, LoD lines, and sample points.
   * You will be prompted to save the results to a new Excel file.

---

## Data Format Guidelines

* Data should be arranged in a single worksheet without mandatory headers.
* Standards and signals can be in any row or column but must align spatially (e.g., same index positions).
* Samples (optional) follow the same alignment rules for name and signal selection.

---

## License

This project is released under the [MIT License](LICENSE).

---

*For questions or issues, please open an issue on the project repository or contact the author.*
