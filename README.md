# Invisible Map Generation

This repository is a refactor and extension of the work done in [occamlab/assistive_apps](https://github.com/occamLab/assistive_apps/tree/summer2018) to generate maps.

## Dependencies
- [g2opy](https://github.com/uoip/g2opy) to work with the graphs.
  - There is an issue with newer versions of Eigen, a dependency of g2opy.
    [This pull request](https://github.com/uoip/g2opy/pull/16) fixes it.
  - If g2o is building for the wrong python version, see [this issue](https://github.com/uoip/g2opy/issues/9).
- Additional Python requirements can be installed using the included requirements.txt

## Source Files
- `map_processing`: Python package containing the files that process the maps using g2o
  - `graph.py` The graph type that uses the python g2o bindings
  - `graph_utils.py`: Contains useful helper functions and types for graphs, such as converting them to a dict of arrays for plotting or integrating measurements into a path.
  - `graph_vertex_edge_classes.py`: Classes used by graph.py for components of the graph
  - `as_graph.py`: Main file that contains functions to convert the raw JSON map files to graphs that can be processed
  - `firebase_manager.py`: File with the FirebaseManager class that handles interactions with Firebase (download, upload)
  - `graph_manager.py`: Main file with the GraphManager class that handles the optimization of the map
- `run_scripts`: Python package containing the files that can be run to either process, optimize, or evaluate maps and map parameters
  - `graph_manager_user.py`: Script to manually download, process, and visualize maps
  - `process_graphs.py`: Script to run a continuous listener that downloads new maps added to Firebase, processes, and uploads them
  - `optimize_weights.py`: Script to run a genetic algorithm optimization on the weights used for g2o
  - `correlate_matrics.py`: Script to find the correlation between different graph evaluation metrics
  - `visualize_chi2s.py`: Script to visualize the results of a weights sweep using the chi2 error as the metric

## Additional Directories
- `/archive`: Code that has been replaced or deprecated
- `/converted-data`, `/data`: Old data files for previous map types
- `/expectation_maximization`: Previously used maximization model
- `/g2opy_setup`: Setup help and script for g2opy
- `/img`: Pictures for this README
- `/notebooks`: Jupyter notebooks
- `/saved_chi2_sweeps`, `/saved_sweep_results`: Saved data files for parameter sweeps

## `GraphManager.py` Manual Usage

The `graph_manager_user` script and `GraphManager` class in `GraphManager.py` provides multiple capabilities:

- Acquiring and caching unprocessed maps from the Firebase database.
- Performing standard graph optimization with plotting capabilities.
- Performing a graph optimization comparison routine (see help for the `-c` flag or, for more detail, documentation 
  of the `GraphManager.compare_weights` instance method).

The script is operated through command line arguments. To see the help message, run:

```
python3 -m run_scripts.graph_manager_user -h
```

### Example usage

1. Acquire and cache unprocessed maps:

```
python3 -m run_scripts.graph_manager_user -f
```

This invokes an infinite loop that listens to the database request, so it will need to be manually quit with Ctrl+C.
Also note that this only pulls graphs added after it is run (the code should ideally be modified to download all uncached maps).
Maps can be manually downloaded into the .cache directory if necessary.

2. Run standard graph optimization routine (with visualization turned on) with any maps matching the `glob` pattern (from the `.cache/` directory) of `unprocessed_maps/**/*Marion*`: 

```
python3 -m run_scripts.graph_manager_user -p "unprocessed_maps/**/*Marion*" -v
```

3. Run the optimization comparison routine:

```
python3 -m run_scripts.graph_manager_user -p "unprocessed_maps/**/*Marion*" -v -c
```

## Automatic Map Processing
The `process_graphs` script allows new maps to be downloaded automatically from Firebase, optimized, and uploaded to Firebaese again.
This is intended  to be the primary script that constantly runs on a backend server to process and upload maps that users of
InvisbleMapCreator make for users of InvisibleMap to navigate.

The script can be run through command line with:
```
python3 -m run_scripts.process_graphs
```

## TODOS
- Continue finding ways to evaluate the quality of optimized maps.
  - Consider ways of obtaining ground truth data
- Add more ways to consolidate paths in the map to make navigation more efficient
  - Currently, only direct intersections are handled
  - Consider detecting points that have no obstructions between (e.g. connect odometry points that are on different sides of a hallway).
