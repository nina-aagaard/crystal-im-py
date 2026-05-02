from crystal_tools.imports import *
from scipy.optimize import curve_fit

def detect_equilibration(df, window=10, deriv_threshold=0.1):
    # Smooth first (when enough data exists), then differentiate
    if len(df['time']) > 2 * window:
        smooth_temps = df['temp'].rolling(window=window, center=True).mean().bfill().ffill()
    else:
        smooth_temps = df['temp']

    dt = np.diff(df['time'])
    dT = np.diff(smooth_temps)
    smooth_deriv = dT / dt

    for i in range(len(smooth_deriv)):
        if i + window <= len(smooth_deriv):
            window_slice = smooth_deriv[i:i+window]
            if np.max(np.abs(window_slice)) < deriv_threshold:
                return i + window // 2

    return None  # equilibration not detected

def sinusoidal_model(t, T_eq, A, omega, phi):
    """Sinusoidal oscillation model around equilibrium"""
    return T_eq + A * np.sin(omega * t + phi)

def analyze_temperature_series(df, target_temp=None):
    """
    Hybrid approach: detect equilibration, then fit sinusoidal model
    Returns equilibrium temperature and fit parameters
    """
    # Convert to numpy arrays if needed
    time = np.array(df['time'])
    temp = np.array(df['temp'])
    
    # Step 1: Detect equilibration point
    eq_idx = detect_equilibration(df)
    
    # Make sure we have enough data
    if eq_idx is None or eq_idx >= len(time) - 10:
        eq_idx = int(len(time) * 0.3)
    
    # Extract post-equilibration data
    time_eq = time[eq_idx:]
    temp_eq = temp[eq_idx:]
    
    # Adjust time to start from 0 for fitting
    time_fit = time_eq - time_eq[0]
    
    # Step 2: Fit sinusoidal model
    # Initial guesses
    T_eq_guess = np.mean(temp_eq)
    A_guess = (np.max(temp_eq) - np.min(temp_eq)) / 2
    
    # Estimate frequency from data using FFT
    if len(temp_eq) > 5:
        temp_centered = temp_eq - T_eq_guess
        fft = np.fft.fft(temp_centered)
        freqs = np.fft.fftfreq(len(temp_eq), d=np.mean(np.diff(time_fit)) if len(time_fit) > 1 else 1)
        
        # Find dominant frequency (excluding DC component)
        fft_mag = np.abs(fft[1:len(fft)//2])
        if len(fft_mag) > 0:
            peak_freq_idx = np.argmax(fft_mag) + 1
            omega_guess = 2 * np.pi * np.abs(freqs[peak_freq_idx])
        else:
            omega_guess = 0.1
    else:
        omega_guess = 0.1
    
    if omega_guess == 0 or not np.isfinite(omega_guess):
        omega_guess = 0.1
    
    phi_guess = 0
    
    try:
        # Fit the model
        popt, pcov = curve_fit(
            sinusoidal_model, 
            time_fit, 
            temp_eq,
            p0=[T_eq_guess, A_guess, omega_guess, phi_guess],
            maxfev=10000,
            bounds=([temp_eq.min()-10, 0, 0, -2*np.pi],
                    [temp_eq.max()+10, temp_eq.max()-temp_eq.min(), 10, 2*np.pi])
        )
        
        T_eq, A, omega, phi = popt
        
        return {
            'equilibrium_temp': T_eq,
            'amplitude': A,
            'omega': omega,
            'phase': phi,
            'eq_index': eq_idx,
            'fit_params': popt,
            'time_fit': time_fit,
            'temp_fit': temp_eq,
            'time_eq_start': time[eq_idx]
        }
    except Exception as e:
        print(f"  Fitting failed for target {target_temp}°C: {e}")
        print(f"  Using simple mean of post-equilibration data")
        # If fitting fails, just return the mean
        return {
            'equilibrium_temp': T_eq_guess,
            'amplitude': A_guess,
            'omega': omega_guess,
            'phase': 0,
            'eq_index': eq_idx,
            'fit_params': None,
            'time_fit': time_fit,
            'temp_fit': temp_eq,
            'time_eq_start': time[eq_idx]
        }

# Define exponential function for fitting
def exp_decay(t, a, b, c):
    """
    Exponential decay function: y = a * exp(-b * t) + c
    
    Parameters:
    - a: amplitude of decay
    - b: decay rate constant
    - c: equilibrium value (asymptote)
    
    Returns the equilibrium value 'c'
    """
    return a * np.exp(-b * t) + c

# Find equilibrated temperature and equilibrium constant for each experiment
def calc_equilibrium(df_list, temp_model='sinusoidal'):
    """
    Parameters
    ----------
    df_list : list of DataFrames
        Each DataFrame must contain 'time', 'temp', and 'q' columns.
    temp_model : str, optional
        Temperature fitting model to use. Either 'sinusoidal' (default) for the
        hybrid sinusoidal model, or 'exponential' for the exponential decay model.
    """
    if temp_model not in ('sinusoidal', 'exponential'):
        raise ValueError(f"temp_model must be 'sinusoidal' or 'exponential', got '{temp_model}'")

    eq_temps = []
    eq_consts = []
    
    for i, df in enumerate(df_list):
        print(f"Processing Experiment {i + 1}...")
        try:
            time = df['time'].values
            temp = df['temp'].values
            q = df['q'].values
            
            # --- Temperature fitting ---
            if temp_model == 'sinusoidal':
                temp_result = analyze_temperature_series(df, target_temp=None)
                equilibrium_temp = temp_result['equilibrium_temp']
                eq_index = temp_result['time_eq_start']
            else:  # exponential
                popt_temp, _ = curve_fit(exp_decay, time, temp,
                                         p0=[temp[0] - temp[-1], 0.1, temp[-1]],
                                         bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                                         maxfev=10000)
                equilibrium_temp = popt_temp[2]
                temp_result = None
                eq_index = df['time'].min()  # no induction period detected; shade nothing
            eq_temps.append(equilibrium_temp)
            
            # --- Reaction quotient: exponential decay ---
            popt_q, _ = curve_fit(exp_decay, time, q, 
                                  p0=[q[0] - q[-1], 0.1, q[-1]],
                                  bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                                  maxfev=10000)
            equilibrium_constant = popt_q[2]
            eq_consts.append(equilibrium_constant)
            
            print(f"  DataFrame {i + 1}: T_eq = {equilibrium_temp:.4f}, K_eq = {equilibrium_constant:.4f}")

            # --- Plotting ---
            figure, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Temperature plot
            axes[0].plot(df['time'], df['temp'], color='tab:orange')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Temperature (°C)')
            
            # Overlay fit curve if sinusoidal model was used and fit succeeded
            if temp_model == 'sinusoidal' and temp_result['fit_params'] is not None:
                t_plot = temp_result['time_fit']
                fit_curve = sinusoidal_model(t_plot, *temp_result['fit_params'])
                axes[0].plot(t_plot + eq_index, fit_curve, color='tab:red',
                             ls='--', lw=1.5, label='Sinusoidal fit')
            elif temp_model == 'exponential':
                fit_curve = exp_decay(time, *popt_temp)
                axes[0].plot(time, fit_curve, color='tab:red',
                             ls='--', lw=1.5, label='Exponential fit')
            axes[0].axhline(equilibrium_temp, color='tab:orange', ls='--', label='Equilibrated temperature')
            axes[0].legend()

            # Reaction quotient plot
            axes[1].plot(df['time'], df['q'], color='tab:purple')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Reaction Quotient Q')
            q_fit_curve = exp_decay(time, *popt_q)
            axes[1].plot(time, q_fit_curve, color='tab:red',
                         ls='--', lw=1.5, label='Exponential fit')
            axes[1].axhline(equilibrium_constant, color='tab:purple', ls='--', alpha=0.6,
                            label='Equilibrium constant')
            axes[1].legend()

        except Exception as e:
            print(f"  DataFrame {i + 1}: Failed to fit - {str(e)}")
            eq_temps.append(np.nan)
            eq_consts.append(np.nan)

    print("\nProcessing complete!")
    print(f"Total experiments processed: {len(df_list)}")
    print(f"Results stored in eq_temps and eq_consts lists")
    return eq_temps, eq_consts