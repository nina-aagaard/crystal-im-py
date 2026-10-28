from sympy.core import expr, symbol
from scipy.optimize import brentq

from crystal_tools.imports import *
import sympy as sp
R = 8.31446261815324  # J/(mol·K), gas constant

def thermodynamics_ext(A, B, c, sA, sB, sc):
    """
    Return ΔH, ΔS, ΔCp (all in J/mol or J/(mol·K)) at a reference temperature.
    """
    delta_Cp  = R * B
    # Use sympy to keep ΔH and ΔS as functions of T for the extended van't Hoff model
    delta_H   = R * (A + B * sp.Symbol('T'))   # J/mol
    delta_S   = R * (B * sp.log(sp.Symbol('T')) + B + c)  # J/(mol·K)

    # Uncertainties for thermodynamic quantities:
    s_delta_Cp  = R * sB
    s_delta_H   = R * np.sqrt(sA**2 + (sp.Symbol('T') * sB)**2)
    s_delta_S   = R * np.sqrt((sp.log(sp.Symbol('T')) * sB)**2 + sB**2 + sc**2)
    s_delta_G   = np.sqrt(s_delta_H**2 + (sp.Symbol('T') * s_delta_S)**2)
    return delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G

def thermodynamics_lin(A, c, sA, sc):
    """
    Return ΔH, ΔS, ΔCp (all in J/mol or J/(mol·K)).
    ΔCp is identically zero under the linear van't Hoff assumption.
    """
    delta_H  = R * A    # J/mol  (divide by 1000 for kJ/mol if preferred)
    delta_S  = R * c    # J/(mol·K)
    delta_Cp = 0.0      # linear model assumes temperature-independent ΔH

    # Uncertainties for thermodynamic quantities:
    s_delta_H  = R * sA
    s_delta_S  = R * sc
    s_delta_Cp = 0.0
    s_delta_G  = np.sqrt(s_delta_H**2 + (sp.Symbol('T') * s_delta_S)**2)
    return delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G

def zero_crossing(expr, T_sym, T_range=(200, 1000), n_scan=500):
    """Find zero crossing numerically by scanning for sign change, then refining with brentq."""
    f = sp.lambdify(T_sym, expr, 'numpy')
    T_vals = np.linspace(T_range[0], T_range[1], n_scan)
    f_vals = f(T_vals)
    
    # Find where sign changes
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    
    if len(sign_changes) == 0:
        return None  # no crossing found in range
    
    # Use the first sign change to bracket brentq
    T_low = T_vals[sign_changes[0]]
    T_high = T_vals[sign_changes[0] + 1]
    
    try:
        T_zero = brentq(f, T_low, T_high)
        return T_zero - 273.15
    except ValueError:
        return None

# thermodynamic profiles from van't Hoff regression data
def vant_hoff_thermo(eq_temps, reg_coeffs, vh_fit_model='nonlinear'):
    print("Function entered")  # diagnostic print statement
    """
    Inputs:
    eq_temps: list of equilibrated temperatures calculated using calc_equilibrium function
    reg_coeffs: regression coefficients from vant_hoff_plot function (either linear or non-linear)
    vh_fit_model: van't Hoff equation model, can either use linear van't Hoff equation or 3-variable truncation of extended van't Hoff
        (accounts for change in heat capacity during transformation)

    Outputs:
    Thermodynamic profiles (ΔH, ΔS, ΔCp) as functions of temperature
    """
    if vh_fit_model == 'nonlinear':
        A, B, c, sA, sB, sc, r2 = reg_coeffs
        delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G = thermodynamics_ext(A, B, c, sA, sB, sc)
        
        # Create plots of ΔH, ΔS, and ΔG as functions of temperature
        figure, axes = plt.subplots(3, 1, figsize=(8, 12))
        
        T_range = max(eq_temps) - min(eq_temps)
        # Convert eq_temps to Kelvin for plotting and extend range by 25% on either side
        T_vals = np.linspace((min(eq_temps) + 273.15) - 0.25 * T_range, max(eq_temps) + 273.15 + 0.25 * T_range, 200)

        H_vals = np.array([delta_H.subs(sp.Symbol('T'), T) for T in T_vals])
        sH_vals = np.array([s_delta_H.subs(sp.Symbol('T'), T) for T in T_vals])
        S_vals = np.array([delta_S.subs(sp.Symbol('T'), T) for T in T_vals])
        sS_vals = np.array([s_delta_S.subs(sp.Symbol('T'), T) for T in T_vals])
        G_vals = H_vals - T_vals * S_vals
        sG_vals = np.sqrt(sH_vals**2 + (T_vals * sS_vals)**2)

        # First plot: ΔH vs T
        axes[0].plot(T_vals-273.15, H_vals/1000, color='blue')
        axes[0].fill_between(T_vals-273.15, (H_vals-sH_vals)/1000, (H_vals+sH_vals)/1000, color='blue', alpha=0.2)
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('ΔH° (kJ/mol)')
        
        # Second plot: ΔS vs T
        axes[1].plot(T_vals-273.15, S_vals, color='orange')
        axes[1].fill_between(T_vals-273.15, S_vals-sS_vals, S_vals+sS_vals, color='orange', alpha=0.2)
        axes[1].set_xlabel('Temperature (°C)')
        axes[1].set_ylabel('ΔS° (J/(mol·K))')

        # Third plot: ΔG vs T
        axes[2].plot(T_vals-273.15, G_vals/1000, color='green')
        axes[2].fill_between(T_vals-273.15, (G_vals-sG_vals)/1000, (G_vals+sG_vals)/1000, color='green', alpha=0.2)
        axes[2].set_xlabel('Temperature (°C)')
        axes[2].set_ylabel('ΔG° (kJ/mol)')

        plt.tight_layout()

        # Print the equations for ΔH, ΔS, and ΔG as functions of temperature
        print(f"ΔH(T) = {delta_H} ± {s_delta_H} J/mol")
        print(f"ΔS(T) = {delta_S} ± {s_delta_S} J/(mol·K)")
        print(f"ΔG(T) = {delta_H} J/mol - T * {delta_S} J/(mol·K) ± {s_delta_G} J/mol")
        
        # Print the value for ΔCp (constant for the extended van't Hoff model)
        print(f"ΔCp = {delta_Cp:.2f} ± {s_delta_Cp:.2f} J/(mol·K)")

        # Report values and uncertainties for ΔH, ΔS, and ΔG at 298.15 K
        H_298 = delta_H.subs(sp.Symbol('T'), 298.15)
        S_298 = delta_S.subs(sp.Symbol('T'), 298.15)
        G_298 = H_298 - 298.15 * S_298
        s_H_298 = s_delta_H.subs(sp.Symbol('T'), 298.15)
        s_S_298 = s_delta_S.subs(sp.Symbol('T'), 298.15)
        s_G_298 = s_delta_G.subs(sp.Symbol('T'), 298.15)
        print(" ") # add a blank line for better readability
        print(f"At 25°C (298.15 K): ΔH = {H_298:.2f} ± {s_H_298:.2f} J/mol \nΔS = {S_298:.2f} ± {s_S_298:.2f} J/(mol·K) \nΔG = {G_298:.2f} ± {s_G_298:.2f} J/mol")

        # Report values and uncertainities for ΔH, ΔS, and ΔG at 64 °C
        H_337 = delta_H.subs(sp.Symbol('T'), 337.15)
        S_337 = delta_S.subs(sp.Symbol('T'), 337.15)
        G_337 = H_337 - 337.15 * S_337
        s_H_337 = s_delta_H.subs(sp.Symbol('T'), 337.15)
        s_S_337 = s_delta_S.subs(sp.Symbol('T'), 337.15)
        s_G_337 = s_delta_G.subs(sp.Symbol('T'), 337.15)
        print(" ") # add a blank line for better readability
        print(f"At 64°C (337.15 K): ΔH = {H_337:.2f} ± {s_H_337:.2f} J/mol \nΔS = {S_337:.2f} ± {s_S_337:.2f} J/(mol·K) \nΔG = {G_337:.2f} ± {s_G_337:.2f} J/mol")

        T = sp.Symbol('T')
        H_zero_cross = zero_crossing(delta_H, T)
        S_zero_cross = zero_crossing(delta_S, T)
        G_zero_cross = zero_crossing(delta_H - T * delta_S, T)
        
        print(" ") # add a blank line for better readability
        
        print(f"Zero-crossing temperatures:")
        print(f"  ΔH = {f'{H_zero_cross:.2f} °C' if H_zero_cross is not None else 'No crossing in range'}")
        print(f"  ΔS = {f'{S_zero_cross:.2f} °C' if S_zero_cross is not None else 'No crossing in range'}")
        print(f"  ΔG = {f'{G_zero_cross:.2f} °C' if G_zero_cross is not None else 'No crossing in range'}")

    elif vh_fit_model == 'linear':
        A, c, sA, sc, r2 = reg_coeffs
        delta_H, delta_S, delta_Cp, s_delta_H, s_delta_S, s_delta_Cp, s_delta_G = thermodynamics_lin(A, c)

        # Print constant ΔH, ΔS, and ΔCp values
        print(f"ΔH: {delta_H:.2f} ± {s_delta_H:.2f} J/mol")
        print(f"ΔS: {delta_S:.2f} ± {s_delta_S:.2f} J/(mol·K)")
        print(f"ΔCp: {delta_Cp:.2f} J/(mol·K)") # error is irrelevant for the linear model because delta_Cp is assumed to be 0

        # Create a plot of ΔG as a function of temperature
        T_range = max(eq_temps) - min(eq_temps)
        T_vals = np.linspace((min(eq_temps) + 273.15) - 0.25 * T_range, max(eq_temps) + 273.15 + 0.25 * T_range, 200)
        G_vals = np.array([delta_H - T * delta_S for T in T_vals])
        s_G_vals = np.array([s_delta_G.subs(sp.Symbol('T'), T) for T in T_vals])

        plt.plot(T_vals-273.15, G_vals/1000, color='green')
        plt.fill_between(T_vals-273.15, (G_vals-s_G_vals)/1000, (G_vals+s_G_vals)/1000, color='green', alpha=0.2)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('ΔG° (kJ/mol)')
        plt.tight_layout()

        # Print the equation for ΔG as a function of temperature
        print(f"ΔG(T) = {delta_H:.2f} J/mol - T * {delta_S:.2f} J/(mol·K) ± {s_delta_G.subs(sp.Symbol('T')):.2f} J/mol")
        
        # Print the value for ΔG at 298.15 K
        G_298 = delta_H - 298.15 * delta_S
        print(f"ΔG at 25°C (298.15 K): {G_298:.2f} ± {s_delta_G.subs(sp.Symbol('T'), 298.15):.2f} J/mol")

        # Print the value for ΔG at 64 °C (337.15 K)
        G_337 = delta_H - 337.15 * delta_S
        print(f"ΔG at 64°C (337.15 K): {G_337:.2f} ± {s_delta_G.subs(sp.Symbol('T'), 337.15):.2f} J/mol")

        # Print the zero-crossing temperature for ΔG (if it exists)
        G_zero_cross = (delta_H / delta_S) - 273.15 # convert to °C
        print(f"Zero-crossing temperature for ΔG: {G_zero_cross:.2f} °C")

    else: raise ValueError(f"vh_fit_model must be 'linear' or 'nonlinear', got '{vh_fit_model}'")