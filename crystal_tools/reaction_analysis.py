from crystal_tools.imports import *

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