import pytest
import numpy as np
import tkinter as tk
from scipy.optimize import curve_fit
from ELISA_GUI_Main import ELISAGUI

# Helper to create a hidden GUI instance
@ pytest.fixture(scope="module")
def gui():
    root = tk.Tk()
    root.withdraw()
    gui = ELISAGUI(root)
    yield gui
    root.destroy()

@ pytest.mark.parametrize("params", [
    (1.0, 1.2, 5.0, 0.5),
    (2.0, 0.8, 10.0, 1.0)
])
def test_4pl_fit_and_inverse(gui, params):
    # Generate synthetic standard data
    A, B, C, D = params
    x = np.linspace(1, 100, 20)
    y = gui.four_param_logistic(x, A, B, C, D)
    # Fit back
    popt, _ = curve_fit(gui.four_param_logistic, x, y, maxfev=10000)
    # Assert parameters recovered within tolerance
    assert pytest.approx(A, rel=1e-2) == popt[0]
    assert pytest.approx(B, rel=1e-2) == popt[1]
    assert pytest.approx(C, rel=1e-2) == popt[2]
    assert pytest.approx(D, rel=1e-2) == popt[3]
    # Test inverse
    y_test = gui.four_param_logistic( C*2, *popt)
    x_inv = gui.inverse_4pl(y_test, *popt)
    assert pytest.approx(C*2, rel=1e-2) == x_inv

@ pytest.mark.parametrize("params", [
    (1.0, 1.2, 5.0, 0.5, 1.5),
    (2.0, 0.8, 10.0, 1.0, 2.0)
])
def test_5pl_fit(gui, params):
    # Generate synthetic standard data
    A, B, C, D, E = params
    x = np.linspace(1, 100, 20)
    y = gui.five_param_logistic(x, A, B, C, D, E)
    # Fit back
    popt, _ = curve_fit(gui.five_param_logistic, x, y, maxfev=10000)
    # Assert parameters recovered
    assert pytest.approx(A, rel=1e-2) == popt[0]
    assert pytest.approx(B, rel=1e-2) == popt[1]
    assert pytest.approx(C, rel=1e-2) == popt[2]
    assert pytest.approx(D, rel=1e-2) == popt[3]
    assert pytest.approx(E, rel=1e-2) == popt[4]

@ pytest.mark.parametrize("m,b", [(2.0, 1.0), (-1.5, 0.5)])
def test_linear_and_log_linear(gui, m, b):
    # Linear
    x = np.linspace(0, 10, 20)
    y_lin = gui.linear_model(x, m, b)
    # Fit back using numpy polyfit
    p_lin = np.polyfit(x, y_lin, 1)
    assert pytest.approx(m, rel=1e-6) == p_lin[0]
    assert pytest.approx(b, rel=1e-6) == p_lin[1]

    # Log-Linear
    # ensure positive
    x2 = np.linspace(1, 10, 20)
    y_log = gui.log_linear_model(x2, m, b)
    # Fit back on logs
    logc, logs = np.log(x2), np.log(y_log)
    p_log = np.polyfit(logc, logs, 1)
    assert pytest.approx(m, rel=1e-6) == p_log[0]
    assert pytest.approx(b, rel=1e-6) == p_log[1]

def test_sample_extrapolation(gui):
    # test sample extrapolation for 4PL
    A,B,C,D = (1.0, 1.0, 10.0, 0.0)
    x = np.array([5.0, 20.0])
    y = gui.four_param_logistic(x, A, B, C, D)
    # Inverse
    inv = [gui.inverse_4pl(val, A, B, C, D) for val in y]
    assert pytest.approx(x[0], rel=1e-6) == inv[0]
    assert pytest.approx(x[1], rel=1e-6) == inv[1]

# Test competitive ELISA setting too
@pytest.mark.parametrize("method", ["4PL", "5PL", "Linear", "Log-Linear"])
def test_competitive_elisa_flag(gui, method):
    # Just set the assay_type and method, ensure attribute changes
    gui.assay_type.set("competitive")
    gui.analysis_method.set(method)
    assert gui.assay_type.get() == "competitive"
    assert gui.analysis_method.get() == method
