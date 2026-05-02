from crystal_tools.imports import *

# van't Hoff non-linear regression functions
def fit_ext_vanthoff(x_inv_T, x_lnT, y_lnK):
    """
    Fit the ln(T) form of the extended van't Hoff equation by OLS:

    ln K  =  -A/T  +  B·ln(T)  +  c

    which is linear in parameters [A, B, c], so we use numpy lstsq.

    OLS fit of  ln K = -A/T + B·ln(T) + c
    Design matrix columns: [-1/T, ln(T), 1]
    Returns (A, B, c, r_squared).

    R² is computed as 1 - SS_res / SS_tot, the standard coefficient of
    determination for the fit of ln(K) vs. the two predictors.

    Thermodynamic identities (ΔCp assumed constant):
    ΔCp   =  R · B
    ΔH(T) =  R · (A  +  B · T)       [from d(lnK)/d(1/T) = -ΔH/R]
    ΔS(T) =  R · (B·ln(T) + B + c)   [from ΔG = ΔH - TΔS = -RT·lnK]

    Reference: Clarke & Glew (1966) Trans. Faraday Soc. 62, 539–547;
        Grant & Higuchi (1984) Int. J. Pharmaceutics 18.

    """
    X = np.column_stack([-x_inv_T,             # coefficient A  (note sign: col = -1/T)
                          x_lnT,               # coefficient B
                          np.ones_like(x_inv_T)])  # constant c
    result = np.linalg.lstsq(X, y_lnK, rcond=None)
    A, B, c = result[0]
    y_pred  = X @ np.array([A, B, c])
    ss_res  = np.sum((y_lnK - y_pred) ** 2)
    ss_tot  = np.sum((y_lnK - y_lnK.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot
    return A, B, c, r2

def thermodynamics_ext(A, B, c, T_ref):
    """
    Return ΔH, ΔS, ΔCp (all in J/mol or J/(mol·K)) at a reference temperature.
    """
    delta_Cp  = R * B
    delta_H   = R * (A + B * T_ref)   # J/mol  (use kJ/mol by /1000 if preferred)
    delta_S   = R * (B * np.log(T_ref) + B + c)  # J/(mol·K)
    return delta_H, delta_S, delta_Cp

# van't Hoff linear regression functions
def fit_lin_vanthoff(x_inv_T, y_lnK):
    """
    Fit the standard (linear) van't Hoff equation by OLS:
        ln K  =  -ΔH/R · (1/T)  +  ΔS/R
    which is linear in parameters [A, c], so we use numpy lstsq.
    OLS fit of  ln K = -A/T + c
    Design matrix columns: [-1/T, 1]
    Returns (A, c, r_squared).
    R² is computed as 1 - SS_res / SS_tot, the standard coefficient of
    determination for the fit of ln(K) vs. 1/T.
    Thermodynamic identities (ΔCp = 0 assumed):
        ΔH  =  R · A        [from d(lnK)/d(1/T) = -ΔH/R, constant with T]
        ΔS  =  R · c        [from ln K = -ΔH/RT + ΔS/R]
        ΔCp =  0            [implicit assumption of the linear model]
    Reference: van't Hoff (1884); Clarke & Glew (1966) Trans. Faraday Soc. 62, 539–547.
    """
    X = np.column_stack([-x_inv_T,                  # coefficient A  (note sign: col = -1/T)
                          np.ones_like(x_inv_T)])    # constant c
    result = np.linalg.lstsq(X, y_lnK, rcond=None)
    A, c = result[0]
    y_pred = X @ np.array([A, c])
    ss_res = np.sum((y_lnK - y_pred) ** 2)
    ss_tot = np.sum((y_lnK - y_lnK.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    return A, c, r2

def thermodynamics_lin(A, c):
    """
    Return ΔH, ΔS, ΔCp (all in J/mol or J/(mol·K)).
    ΔCp is identically zero under the linear van't Hoff assumption.
    """
    delta_H  = R * A    # J/mol  (divide by 1000 for kJ/mol if preferred)
    delta_S  = R * c    # J/(mol·K)
    delta_Cp = 0.0      # linear model assumes temperature-independent ΔH
    return delta_H, delta_S, delta_Cp

# van't Hoff plot function
def vant_hoff_plot(eq_temps, eq_consts, temp_fit_model='exponential', vh_fit_model='nonlinear'):
    """
    Inputs:
    eq_temps: list of equilibrated temperatures calculated using calc_equilibrium function
    eq_consts: list of equilibrium constants calculated using calc_equilibrium function
    temp_fit_model: equilibration model for fitting temperature (default is exponential, can also use a hybrid sinusoidal model
        with induction period and then oscillation (better for const. temp, called using 'sinusoidal'))
    vh_fit_model: van't Hoff equation model, can either use linear van't Hoff equation or 3-variable truncation of extended van't Hoff
        (accounts for change in heat capacity during transformation)

    Outputs:
    van't Hoff plot
    van't Hoff regression parameters (linear or non-linear)
    Thermodynamic parameters (constant or functions of temp)
    """
    # Get parameters for van't Hoff plot
    vh_df = pd.DataFrame(list(zip(eq_temps, eq_consts)), columns=['temp', 'k_eq'])
    vh_df['temp_k'] = vh_df['temp'] + 273.15
    vh_df['1/T'] = 1/vh_df['temp_k']
    vh_df['ln(T)'] = np.log(vh_df['temp_k']) # need for non-linear expansion
    vh_df['ln(keq)'] = np.log(vh_df['k_eq'])

    # Plot van't Hoff plot
    plt.scatter(vh_df['1/T'], vh_df['ln(keq)'])
    plt.xlabel('1/T (1/K)')
    plt.ylabel('ln(K$_{eq}$)')

    # Add van't Hoff regression
    if vh_fit_model == 'nonlinear':
        A, B, c, r2 = fit_ext_vanthoff(vh_df['1/T'], vh_df['ln(T)'], vh_df['ln(keq)'])

        # Add regression to plot
        x_fit_inv = np.linspace(vh_df['1/T'].min(), vh_df['1/T'].max(), 200)
        T_fit     = 1.0 / x_fit_inv          # back to Kelvin
        y_fit     = -A * x_fit_inv + B * np.log(T_fit) + c
        plt.plot(x_fit_inv, y_fit, linestyle='-', linewidth=2, alpha=0.8)
        
    elif vh_fit_model == 'linear':
        A, c, r2 = fit_lin_vanthoff(vh_df['1/T'], vh_df['ln(keq)'])

        # Add regression to plot
        x_fit_inv = np.linspace(vh_df['1/T'].min(), vh_df['1/T'].max(), 200)
        T_fit     = 1.0 / x_fit_inv          # back to Kelvin
        y_fit     = -A * x_fit_inv + c
        plt.plot(x_fit_inv, y_fit, linestyle='-', linewidth=2, alpha=0.8)
        
    else: raise ValueError(f"vh_fit_model must be 'linear' or 'nonlinear', got '{vh_fit_model}'")
    plt.tight_layout()