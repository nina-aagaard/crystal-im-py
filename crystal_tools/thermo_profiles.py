from crystal_tools.imports import *
import sympy as sp
R = 8.31446261815324  # J/(mol·K), gas constant

def thermodynamics_ext(A, B, c):
    """
    Return ΔH, ΔS, ΔCp (all in J/mol or J/(mol·K)) at a reference temperature.
    """
    delta_Cp  = R * B
    # Use sympy to keep ΔH and ΔS as functions of T for the extended van't Hoff model
    delta_H   = R * (A + B * sp.Symbol('T'))   # J/mol
    delta_S   = R * (B * sp.log(sp.Symbol('T')) + B + c)  # J/(mol·K)
    return delta_H, delta_S, delta_Cp

def thermodynamics_lin(A, c):
    """
    Return ΔH, ΔS, ΔCp (all in J/mol or J/(mol·K)).
    ΔCp is identically zero under the linear van't Hoff assumption.
    """
    delta_H  = R * A    # J/mol  (divide by 1000 for kJ/mol if preferred)
    delta_S  = R * c    # J/(mol·K)
    delta_Cp = 0.0      # linear model assumes temperature-independent ΔH
    return delta_H, delta_S, delta_Cp

# thermodynamic profiles from van't Hoff regression data
def vant_hoff_thermo(eq_temps, reg_coeffs, vh_fit_model='nonlinear'):
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
        A, B, c, r2 = reg_coeffs
        delta_H, delta_S, delta_Cp = thermodynamics_ext(A, B, c)

        # Create plots of ΔH, ΔS, and ΔG as functions of temperature
        figure, axes = plt.subplots(3, 1, figsize=(8, 12))
        
        T_range = max(eq_temps) - min(eq_temps)
        # Convert eq_temps to Kelvin for plotting and extend range by 25% on either side
        T_vals = np.linspace((min(eq_temps) + 273.15) - 0.25 * T_range, max(eq_temps) + 273.15 + 0.25 * T_range, 200)

        H_vals = np.array([delta_H.subs(sp.Symbol('T'), T) for T in T_vals])
        S_vals = np.array([delta_S.subs(sp.Symbol('T'), T) for T in T_vals])
        G_vals = H_vals - T_vals * S_vals

        # First plot: ΔH vs T
        axes[0].plot(T_vals-273.15, H_vals/1000, color='blue')
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('ΔH° (kJ/mol)')
        
        # Second plot: ΔS vs T
        axes[1].plot(T_vals-273.15, S_vals, color='orange')
        axes[1].set_xlabel('Temperature (°C)')
        axes[1].set_ylabel('ΔS° (J/(mol·K))')

        # Third plot: ΔG vs T
        axes[2].plot(T_vals-273.15, G_vals/1000, color='green')
        axes[2].set_xlabel('Temperature (°C)')
        axes[2].set_ylabel('ΔG° (kJ/mol)')

        plt.tight_layout()
    
    elif vh_fit_model == 'linear':
        A, c, r2 = reg_coeffs
        delta_H, delta_S, delta_Cp = thermodynamics_lin(A, c)

        # Print constant ΔH, ΔS, and ΔCp values
        print(f"ΔH: {delta_H:.2f} J/mol")
        print(f"ΔS: {delta_S:.2f} J/(mol·K)")
        print(f"ΔCp: {delta_Cp:.2f} J/(mol·K)")

        # Create a plot of ΔG as a function of temperature
        T_range = max(eq_temps) - min(eq_temps)
        T_vals = np.linspace((min(eq_temps) + 273.15) - 0.25 * T_range, max(eq_temps) + 273.15 + 0.25 * T_range, 200)
        G_vals = np.array([delta_H - T * delta_S for T in T_vals])

        plt.plot(T_vals-273.15, G_vals/1000, color='green')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('ΔG° (kJ/mol)')
        plt.tight_layout()

    else: raise ValueError(f"vh_fit_model must be 'linear' or 'nonlinear', got '{vh_fit_model}'")