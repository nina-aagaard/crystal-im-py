from crystal_tools.imports import *
from scipy.interpolate import interp1d


def calculate_channel_intensities(images, time_interval, channels={'green': 1, 'blue': 2}):
    """
    Calculate mean intensities for specified color channels across a list of images.
    
    Parameters:
    -----------
    images : list
        List of image arrays (numpy arrays from skimage)
    time_interval : float, optional
        Time interval between images in seconds (default: 1.94)
    channels : dict, optional
        Dictionary mapping channel names to their indices (default: {'green': 1, 'blue': 2})
        Standard RGB indexing: Red=0, Green=1, Blue=2
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns for time and mean intensity of each channel
        
    Raises:
    -------
    ValueError
        If images list is empty or images don't have the required channels
    """
    if not images:
        raise ValueError("Images list is empty")
    
    # Verify that images have enough channels
    if images[0].ndim < 3:
        raise ValueError("Images must be color images with at least 3 channels")
    
    max_channel_idx = max(channels.values())
    if images[0].shape[2] <= max_channel_idx:
        raise ValueError(f"Images don't have enough channels. Required: {max_channel_idx + 1}, "
                        f"Available: {images[0].shape[2]}")
    
    # Initialize dictionary to store mean intensities for each channel
    channel_means = {channel_name: [] for channel_name in channels.keys()}
    
    # Process images
    for img_array in images:
        for channel_name, channel_idx in channels.items():
            channel_data = img_array[:, :, channel_idx]
            channel_means[channel_name].append(np.mean(channel_data))
    
    # Create time array
    time = np.arange(len(images)) * time_interval
    
    # Create dataframe
    df_data = {'time': time}
    for channel_name, means in channel_means.items():
        df_data[f'mean_{channel_name}'] = means
    
    df = pd.DataFrame(df_data)
    
    print(f"Processed {len(images)} images")
    print(f"Time range: {time[0]:.2f}s to {time[-1]:.2f}s")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def add_temp_data(temp_path, df, time_interval):
    temp_df = pd.read_csv(temp_path)
    time = temp_df['Time (s)'].values
    temperature = temp_df['Temperature (°C)'].values
    interp_func = interp1d(time, temperature, kind='linear', fill_value='extrapolate', bounds_error=False)
    microscopy_times = np.arange(0, time_interval*len(df), time_interval)
    interpolated_temps = interp_func(microscopy_times)
    
    interpolated_df = pd.DataFrame({
        'Time (s)': microscopy_times,
        'Interpolated Temperature (°C)': interpolated_temps
    })

    df['temp'] = interpolated_df['Interpolated Temperature (°C)']

    return df