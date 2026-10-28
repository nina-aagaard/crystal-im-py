import warnings

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import brentq

from crystal_tools.imports import *  # noqa: F401,F403  (keeps any other names your notebooks rely on)

R = 8.31446261815324  # J/(mol·K), gas constant
T_SYM = sp.Symbol('T')


# ---------------------------------------------------------------------------
# Regression with covariance
# ---------------------------------------------------------------------------
def _ols(X, y):
    """OLS fit returning (beta, cov, r2). cov = s² (XᵀX)⁻¹, computed via pinv(X) for stability."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    n, p = X.shape
    dof = n - p
    if dof > 0:
        Xp = np.linalg.pinv(X)
        cov = (ss_res / dof) * (Xp @ Xp.T)   # (XᵀX)⁻¹ = X⁺ X⁺ᵀ
    else:
        cov = np.full((p, p), np.nan)        # exact fit: no uncertainty information
    return beta, cov, r2


def fit_vant_hoff_ext(x_inv_T, x_lnT, y_lnK):
    """
    Extended van't Hoff:  ln K = -A/T + B ln T + c
    Returns (A, B, c, sA, sB, sc, r2, cov), where cov is the 3x3 covariance matrix of (A, B, c).
    """
    X = np.column_stack([-np.asarray(x_inv_T), np.asarray(x_lnT), np.ones_like(x_inv_T, dtype=float)])
    beta, cov, r2 = _ols(X, np.asarray(y_lnK))
    A, B, c = beta
    sA, sB, sc = np.sqrt(np.diag(cov))
    return A, B, c, sA, sB, sc, r2, cov


def fit_vant_hoff_lin(x_inv_T, y_lnK):
    """
    Linear van't Hoff:  ln K = -A/T + c
    Returns (A, c, sA, sc, r2, cov), where cov is the 2x2 covariance matrix of (A, c).
    """
    X = np.column_stack([-np.asarray(x_inv_T), np.ones_like(x_inv_T, dtype=float)])
    beta, cov, r2 = _ols(X, np.asarray(y_lnK))
    A, c = beta
    sA, sc = np.sqrt(np.diag(cov))
    return A, c, sA, sc, r2, cov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _propagate(grad, cov):
    """
    First-order (delta-method) error propagation: sqrt(gᵀ Σ g), with g the gradient of a
    quantity w.r.t. the fitted parameters. grad is a sympy column Matrix (may depend on T).
    """
    C = sp.Matrix(np.asarray(cov, dtype=float).tolist())
    var = (grad.T * C * grad)[0]
    return sp.sqrt(var)  # negative round-off is clipped to 0 numerically in _num()


def _num(expr):
    """Turn a sympy expression in T into a function T -> float ndarray (handles constants)."""
    # Override sqrt so tiny negative variances from round-off become 0 rather than NaN
    safe_sqrt = {'sqrt': lambda x: np.sqrt(np.maximum(x, 0.0))}
    f = sp.lambdify(T_SYM, expr, modules=[safe_sqrt, 'numpy'])

    def func(T):
        T = np.asarray(T, dtype=float)
        return np.broadcast_to(np.asarray(f(T), dtype=float), T.shape).copy()
    return func


# ---------------------------------------------------------------------------
# Thermodynamic expressions
# ---------------------------------------------------------------------------
def thermodynamics_ext(A, B, c, cov):
    """
    Extended van't Hoff model, ln K = -A/T + B ln T + c.

        ΔH(T) = R (A + B T)
        ΔS(T) = R (B ln T + B + c)
        ΔCp   = R B
        ΔG(T) = ΔH - T ΔS = R (A - B T ln T - c T)

    Uncertainties use the full covariance matrix of (A, B, c). Gradients w.r.t. (A, B, c):
        ∇ΔH = R [1, T, 0]
        ∇ΔS = R [0, ln T + 1, 1]
        ∇ΔG = ∇ΔH - T ∇ΔS = R [1, -T ln T, -T]

    Returns sympy expressions (functions of T) for ΔH, ΔS and their 1σ uncertainties,
    plus ΔCp, s_ΔCp (floats) and the 1σ uncertainty of ΔG:
        delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G
    """
    T = T_SYM
    A, B, c = float(A), float(B), float(c)
    cov = np.asarray(cov, dtype=float)

    delta_H = R * (A + B * T)
    delta_S = R * (B * sp.log(T) + B + c)
    delta_Cp = R * B

    gH = R * sp.Matrix([1, T, 0])
    gS = R * sp.Matrix([0, sp.log(T) + 1, 1])
    gG = gH - T * gS

    s_delta_H = _propagate(gH, cov)
    s_delta_S = _propagate(gS, cov)
    s_delta_G = _propagate(gG, cov)
    s_delta_Cp = R * float(np.sqrt(cov[1, 1]))
    return delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G


def thermodynamics_lin(A, c, cov):
    """
    Linear van't Hoff model, ln K = -A/T + c.

        ΔH = R A,  ΔS = R c,  ΔCp = 0,  ΔG(T) = R (A - c T)

    cov is the 2x2 covariance matrix of (A, c). Same return layout as thermodynamics_ext
    (ΔH, ΔS and their uncertainties are constants, returned as sympy numbers).
    """
    T = T_SYM
    A, c = float(A), float(c)
    cov = np.asarray(cov, dtype=float)

    delta_H = sp.Float(R * A)
    delta_S = sp.Float(R * c)
    delta_Cp = 0.0   # linear model assumes temperature-independent ΔH
    s_delta_Cp = 0.0

    gH = R * sp.Matrix([1, 0])
    gS = R * sp.Matrix([0, 1])
    gG = gH - T * gS

    s_delta_H = _propagate(gH, cov)
    s_delta_S = _propagate(gS, cov)
    s_delta_G = _propagate(gG, cov)
    return delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G


# ---------------------------------------------------------------------------
# Zero crossings
# ---------------------------------------------------------------------------
def zero_crossing(expr, T_sym=T_SYM, T_range=(200, 1000), n_scan=500):
    """
    Find the first zero crossing (in °C) of a sympy expression in T (Kelvin) by scanning
    for a sign change and refining with brentq. Returns None if there is no crossing in range.
    """
    f = _num(expr)
    T_vals = np.linspace(T_range[0], T_range[1], n_scan)
    s = np.sign(f(T_vals))

    # NaNs give NaN products, which compare False, so they are never flagged as crossings
    idx = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(idx) == 0:
        return None

    T_low, T_high = T_vals[idx[0]], T_vals[idx[0] + 1]
    try:
        T_zero = brentq(lambda t: float(f(t)), T_low, T_high)
        return T_zero - 273.15
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def vant_hoff_thermo(eq_temps, reg_coeffs, vh_fit_model='nonlinear', T_range_K=(200, 1000)):
    """
    Inputs:
    eq_temps: list of equilibrated temperatures (°C) calculated using calc_equilibrium function
    reg_coeffs: regression output, now including the covariance matrix as the LAST element
        nonlinear: (A, B, c, sA, sB, sc, r2, cov)   -> see fit_vant_hoff_ext
        linear:    (A, c, sA, sc, r2, cov)          -> see fit_vant_hoff_lin
    vh_fit_model: 'linear' (van't Hoff) or 'nonlinear' (3-parameter truncation of the extended
        van't Hoff equation, which accounts for ΔCp)
    T_range_K: temperature window (K) scanned for zero crossings

    Outputs:
    Plots of ΔH, ΔS, ΔG vs T with ±1σ bands (covariance-propagated), printed values at
    25 °C and 64 °C, and zero-crossing temperatures. Returns a dict with the numeric functions.
    """
    T = T_SYM

    if vh_fit_model == 'nonlinear':
        try:
            A, B, c, sA, sB, sc, r2, cov = reg_coeffs
        except ValueError:
            raise ValueError("For 'nonlinear', reg_coeffs must be (A, B, c, sA, sB, sc, r2, cov).")
        delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G = \
            thermodynamics_ext(A, B, c, cov)
    elif vh_fit_model == 'linear':
        try:
            A, c, sA, sc, r2, cov = reg_coeffs
        except ValueError:
            raise ValueError("For 'linear', reg_coeffs must be (A, c, sA, sc, r2, cov).")
        delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G = \
            thermodynamics_lin(A, c, cov)
    else:
        raise ValueError(f"vh_fit_model must be 'linear' or 'nonlinear', got '{vh_fit_model}'")

    if not np.all(np.isfinite(np.asarray(cov, dtype=float))):
        warnings.warn("Covariance matrix contains NaN (n <= number of parameters); "
                      "uncertainty bands will be empty.")

    delta_G = delta_H - T * delta_S
    fH, fS, fG = _num(delta_H), _num(delta_S), _num(delta_G)
    fsH, fsS, fsG = _num(s_delta_H), _num(s_delta_S), _num(s_delta_G)

    # Temperature grid (K), extended 25% beyond the data on either side
    T_span = max(eq_temps) - min(eq_temps)
    T_vals = np.linspace((min(eq_temps) + 273.15) - 0.25 * T_span,
                         (max(eq_temps) + 273.15) + 0.25 * T_span, 200)
    T_C = T_vals - 273.15

    H_vals, sH_vals = fH(T_vals), fsH(T_vals)
    S_vals, sS_vals = fS(T_vals), fsS(T_vals)
    G_vals, sG_vals = fG(T_vals), fsG(T_vals)   # sG includes the ΔH–ΔS covariance

    # --- plots ---
    figure, axes = plt.subplots(3, 1, figsize=(8, 12))
    panels = [
        (H_vals / 1000, sH_vals / 1000, 'blue', 'ΔH° (kJ/mol)'),
        (S_vals, sS_vals, 'orange', 'ΔS° (J/(mol·K))'),
        (G_vals / 1000, sG_vals / 1000, 'green', 'ΔG° (kJ/mol)'),
    ]
    for ax, (y, sy, color, label) in zip(axes, panels):
        ax.plot(T_C, y, color=color)
        ax.fill_between(T_C, y - sy, y + sy, color=color, alpha=0.2)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel(label)
    plt.tight_layout()

    # --- printed results ---
    print(f"ΔH(T) = {sp.N(delta_H, 6)} J/mol")
    print(f"ΔS(T) = {sp.N(delta_S, 6)} J/(mol·K)")
    print(f"ΔG(T) = {sp.N(sp.expand(delta_G), 6)} J/mol")
    print("(1σ uncertainties are propagated with the full parameter covariance matrix)")
    if vh_fit_model == 'nonlinear':
        print(f"ΔCp = {delta_Cp:.2f} ± {s_delta_Cp:.2f} J/(mol·K)")
    else:
        print(f"ΔCp = {delta_Cp:.2f} J/(mol·K) (fixed at 0 in the linear model)")

    for T_K, label in [(298.15, "25°C (298.15 K)"), (337.15, "64°C (337.15 K)")]:
        H, sH = float(fH(T_K)), float(fsH(T_K))
        S, sS = float(fS(T_K)), float(fsS(T_K))
        G, sG = float(fG(T_K)), float(fsG(T_K))
        print(" ")
        print(f"At {label}: ΔH = {H:.2f} ± {sH:.2f} J/mol \n"
              f"ΔS = {S:.2f} ± {sS:.2f} J/(mol·K) \n"
              f"ΔG = {G:.2f} ± {sG:.2f} J/mol")

    # --- zero crossings ---
    crossings = {
        'ΔH': zero_crossing(delta_H, T, T_range_K),
        'ΔS': zero_crossing(delta_S, T, T_range_K),
        'ΔG': zero_crossing(delta_G, T, T_range_K),
    }
    print(" ")
    print("Zero-crossing temperatures:")
    for name, val in crossings.items():
        print(f"  {name} = {f'{val:.2f} °C' if val is not None else 'No crossing in range'}")

    return {
        'H': fH, 'S': fS, 'G': fG,
        'sH': fsH, 'sS': fsS, 'sG': fsG,
        'delta_Cp': delta_Cp, 's_delta_Cp': s_delta_Cp,
        'zero_crossings_C': crossings,
    }
