# crystal-im-py
Colorimetric image analysis for microscopy images of crystals

This repository was developed for data processing of confocal microscopy images of single-crystal-to-single-crystal transitions in solid-state crystals. It is designed to process batches of images exported from Zeiss ZEN microscopy software and extract conversion parameters based on color as a function of time and temperature.

![Image processing workflow for crystal-im-py](crystal-im-py.jpg)

## Getting Started: Setting Up the Repository
If this is your first time using Python or GitHub, follow these instructions to set up the repository. 

First, you'll need to install Git (the tool that lets you download and manage code repositories) by going to git-scm.com/downloads, selecting your operating system, and following the installer prompts; on Windows, accept all default settings. 

Next, install Python by downloading the latest version from python.org/downloads — critically, during installation on Windows, check the box that says "Add Python to PATH" before clicking Install. Once both are installed, open a terminal (on Mac: search for "Terminal" in Spotlight; on Windows: search for "Command Prompt" or "PowerShell" in the Start menu). To download ("clone") this repository to your computer, navigate to the folder where you'd like to save it by typing cd Desktop (or whichever folder you prefer) and pressing Enter, then run git clone https://github.com/nina-aagaard/crystal-im-py.git. Once cloning is complete, move into the new folder by typing cd crystal-im-py and pressing Enter. 

Now install all required libraries in one step by running pip install -r requirements.txt — this automatically installs everything the code needs. Finally, launch Jupyter Notebook by typing jupyter notebook and pressing Enter, which will open an interactive interface in your web browser where you can open and run the analysis notebooks (.ipynb files) by clicking on them.

## Dependencies and Compatibility
All required libraries and their versions are listed in requirements.txt and will be installed automatically when you run pip install -r requirements.txt as described in the setup instructions above. This code was developed and tested on macOS using Python 3.13.5 and should be compatible with any Python 3.13+ installation. The code should also run on Windows with possible minor tweaks. Most commonly, file path formatting differences may require adjusting any hardcoded paths in the notebooks from forward slashes (/) to backslashes (\). If you encounter any issues running the code on Windows, please open an Issue on the GitHub repository page.

## Directory
* **crystal_tools**: Python functions used for image analysis and thermodynamic calculations
  * **data_merging.py**: creates a dataframe of mean channel intensity vs. time based on a list of image arrays and adds temperatures data
  * **image_loading.py**: loads in microscopy image files and creates a list of image arrays
  * **imports.py**: imports common packages for all functions
  * **reaction_analysis.py**: calculates reaction progress and reaction coordinate as functions of time
  * **temperature_fitting.py**: calculates equilibrated temperature based on temperature data and equilibration constant based on reaction quotient using exponential decay functions
  * **thermo_profiles.py**: returns thermodynamic constants or functions based on van't Hoff regression parameters
  * **vant_hoff.py**: creates a van't Hoff plot using equilibrated temperatures and equilibrium constants, then fits either a linear or non-linear van't Hoff equation to data via regression
* **heating_stage_insert_adapter**: .stl files for heating stage adapter for ITO LC cells
* **notebooks**: contains demo notebook with full instructions for running through crystal image analysis and obtaining thermodynamic profiles
* **.gitignore**: list of untracked files
* **README.md**: this text file with instructions for using this repository
* **crystal-im-py.jpg**: abstract image, also shows correct outputs for demo data
* **requirements.txt**: packages needed when using this repository

## How to Use Demo Notebook
Once the repository is set up, you should be able to run the file **demo_notebook.ipynb**. The demo notebook is a comprehensive, step-to-step workup of crystallographic image files from variable temperature van't Hoff experiments with azangulene. Real data is used in this notebook from the folder **demo_data**, and you can compare your outputs to the plots in the figure above to make sure everything is going smoothly.

To launch Jupyter Notebook and run **demo_notebook.ipynb**, navigate to the folder **crystal_im_py** in your command line in Terminal, Command Prompt, or PowerShell (depending on your OS configuration) using the command cd *location of crystal_im_py*. Then type jupyter notebook and press Enter. This should launch Jupyter Notebook in your web browser, where you can navigate to the **notebooks** folder and open **demo_notebook.ipynb**.

## How to Cite
To cite the repository directly:
> Aagaard, N.; Panagis, L. crystal_im_py (Version v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21689911.

If you are citing the research findings associated with this code, please cite the corresponding paper:
> Aagaard, N.; Ragins-Da Rosa, A.; Goh, M.; Panagis, L.; Wiscons, R. A. The ferroelectric Curie temperature of azangulene. *J. Am. Chem. Soc.* (in review).

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.