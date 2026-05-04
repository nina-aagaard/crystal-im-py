from crystal_tools.imports import *

def load_init_image(filepath, init_channel='blue', final_channel='green'):
    """
    Load in initial image and extract initial and final channel intensities for alpha
    
    Parameters:
    -----------
    filepath : directory for initial image (.png format)
    init_channel: initial color channel for alpha parameter (i.e., at alpha=0, this is the dominant channel)
    final_channel: final color channel for alpha parameter (i.e., at alpha=1.0, this is the dominant channel)
    
    Returns:
    --------
    init_0: intensity of initial channel at t=0
    final_0: intensity of final channel at t=0
    """    
    
    # Read image
    image = ski.io.imread(filepath)

    # Extract mean intensity per channel
    red_mean   = np.mean(image[:, :, 0])
    green_mean = np.mean(image[:, :, 1])
    blue_mean  = np.mean(image[:, :, 2])

    if init_channel == 'red':
        init_0 = red_mean
    elif init_channel == 'green':
        init_0 = green_mean
    else: init_0 = blue_mean

    if final_channel == 'red':
        final_0 = red_mean
    elif final_channel == 'green':
        final_0 = green_mean
    else: final_0 = blue_mean
    
    print(f'Initial value for {init_channel} channel = {init_0:.4f}')
    print(f'Initial value for {final_channel} channel = {final_0:.4f}')

    return init_0, final_0

# calculate alpha and q as a function of time

def calculate_rxn_progress(df, init_0, final_0, exp_name, init_channel='blue', final_channel='green'):
    """
    Calculate alpha and reaction quotient q as a function of time
    
    Parameters:
    -----------
    df : dataframe of experiment image data (from image_loading.py and data_analysis.py)
    init_0: intensity of initial channel at t=0 (from load_init_image)
    final_0: intensity of final channel at t=0 (from load_init_image)
    exp_name: experiment name/temperature information (string)
    
    init_channel: initial color channel for alpha parameter (string)
    final_channel: final color channel for alpha parameter (string)
    
    Returns:
    --------
    df['alpha'] = normalized conversion ratio (scale from 0 to 1)
    df['q'] = reaction quotient (scale from 0 to infinity)
    """    

    # calculate alpha and q
    df['alpha'] = 1 - df[f'mean_{init_channel}']/df[f'mean_{final_channel}']*final_0/init_0
    df['q'] = df['alpha']/(1-df['alpha'])

    # create a plot of alpha w.r.t. time
    plt.plot(df['time'], df['alpha'])
    plt.xlabel('Time (s)')
    plt.ylabel('Conversion Ratio α')
    plt.title(f'Conversion Ratio vs. Time for {exp_name}')

    plt.show()