# Invisible Map Generation

This repository is a refactor and extension of the work done in [occamlab/assistive_apps](https://github.com/occamLab/assistive_apps/tree/summer2018) to generate maps.

## Dependencies

- [OCCaM Lab fork of g2opy](https://github.com/occamLab/g2opy) to work with the graphs.
  - Run the setup_g2opy.sh script to set up this dependency
  - The script will automatically source your virtual environment if it exists. If your virtual environment is not in a folder called `./venv/`, then update the `VENV_DIR_REL_PATH` such that it can find your virtual environment. Otherwise, the `g2o.cpython-*.so` file will be installed globally.
- Additional Python requirements can be installed using the included requirements.txt

## Overview

The primary Python packages are:

- `map_processing`: Contains the core optimization functionality.
  - `dataset_generation`: Contains the data set generation functionality.
- `run_scripts`: Contains the scripts that make use of the `map_processing` package.

Refer to each module's docstring for more information. 

### Additional Directories

- `/archive`: Code that has been replaced or deprecated
- `/converted-data`, `/data`: Old data files for previous map types
- `/expectation_maximization`: Previously used maximization model
- `/g2opy_setup`: Setup help and script for g2opy
- `/img`: Pictures for this README
- `/notebooks`: Jupyter notebooks
- `/saved_chi2_sweeps`, `/saved_sweep_results`: Saved data files for parameter sweeps

## Usage

The `run_scripts/graph_manager_user.py` script provides a comprehensive set of capabilities via a command line interface. Execute the script with the `-h` flag to see the help message. Some of its capabilities include:

- Acquiring and caching unprocessed maps from the Firebase database.
- Performing standard graph optimization with plotting capabilities.
- Performing a graph optimization comparison routine (see the the `-c` flag in the help message or, for more detail, documentation 
  of the `GraphManager.compare_weights` instance method).
- Performing a parameter sweep (see the `-s` flag in the help message for more information).

TODO: Add more example usage documentation for the other scripts in `run_scripts/`.

## TODOS

- Continue finding metrics to evaluate optimized map quality.
  - Consider ways of obtaining ground truth data
- Add more ways to consolidate paths in the map to make navigation more efficient
  - Currently, only direct intersections are handled
  - Consider detecting points that have no obstructions between (e.g. connect odometry points that are on different sides of a hallway).
