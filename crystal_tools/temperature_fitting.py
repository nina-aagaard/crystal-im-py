from crystal_tools.imports import *
from scipy.optimize import curve_fit

def detect_equilibration(time, temp, window=10, deriv_threshold=0.1):
    """
    Detect when the system reaches equilibration by analyzing the derivative
    """
    # Calculate smoothed derivative
    dt = np.diff(time)
    dT = np.diff(temp)
    dT_dt = dT / dt
    
    # Smooth the derivative using a rolling window
    if len(dT_dt) < window:
        return int(len(time) * 0.3)
    
    smooth_deriv = np.convolve(dT_dt, np.ones(window)/window, mode='valid')
    
    # Find where the absolute derivative becomes consistently small
    for i in range(len(smooth_deriv)):
        if i + window < len(smooth_deriv):
            window_slice = smooth_deriv[i:i+window]
            if np.max(np.abs(window_slice)) < deriv_threshold:
                return i + window // 2 + window // 2
    
    # Fallback: use 30% of data
    return int(len(time) * 0.3)

def sinusoidal_model(t, T_eq, A, omega, phi):
    """Sinusoidal oscillation model around equilibrium"""
    return T_eq + A * np.sin(omega * t + phi)

def analyze_temperature_series(time, temp, target_temp=None):
    """
    Hybrid approach: detect equilibration, then fit sinusoidal model
    Returns equilibrium temperature and fit parameters
    """
    # Convert to numpy arrays if needed
    time = np.array(time)
    temp = np.array(temp)
    
    # Step 1: Detect equilibration point
    eq_idx = detect_equilibration(time, temp)
    
    # Make sure we have enough data
    if eq_idx >= len(time) - 10:
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