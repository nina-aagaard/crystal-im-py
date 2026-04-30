from crystal_tools.imports import *

def load_images_from_folder(folder_path, max_images=None, file_extension='.png'):
    """
    Load images from a specified folder path, sorted by timepoint.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing images (supports ~ expansion)
    max_images : int or None, optional
        Maximum number of images to load. If None, loads all images (default: None)
    file_extension : str, optional
        File extension to filter (default: '.png')
    
    Returns:
    --------
    images : list
        List of loaded images as numpy arrays
    filenames : list
        List of filenames that were loaded
    
    Raises:
    -------
    FileNotFoundError
        If the specified path does not exist
    ValueError
        If no images with the specified extension are found
    """
    # Expand user path and normalize
    expanded_path = os.path.expanduser(folder_path)
    
    # Check if the path exists
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(f"Path does not exist: {expanded_path}")
    
    print(f"Path exists: {expanded_path}")
    
    # Get all files with specified extension in the folder
    image_files = [f for f in os.listdir(expanded_path) 
                   if f.endswith(file_extension)]
    
    if not image_files:
        raise ValueError(f"No {file_extension} files found in {expanded_path}")
    
    print(f"Found {len(image_files)} {file_extension} files")
    
    # Function to extract timepoint from filename
    def extract_timepoint(filename):
        """Extract timepoint number from filename (e.g., 't98' -> 98)"""
        match = re.search(r't(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            # If no timepoint found, return a large number to put it at the end
            print(f"Warning: Could not find timepoint in filename: {filename}")
            return float('inf')
    
    # Sort files by timepoint
    image_files_sorted = sorted(image_files, key=extract_timepoint)
    
    # Load images up to max_images (or all if max_images is None)
    images = []
    files_to_load = image_files_sorted[:max_images] if max_images is not None else image_files_sorted
    
    for filename in files_to_load:
        img_path = os.path.join(expanded_path, filename)
        img = io.imread(img_path)
        images.append(img)
    
    print(f"Loaded {len(images)} images")
    
    return images, files_to_load