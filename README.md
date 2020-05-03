# Invisible Map Generation

This repository is a refactor and extension of the work done in [occamlab/assistive_apps](https://github.com/occamLab/assistive_apps/tree/summer2018) to generate maps.

## Creating a map
1. Use the app in ARKit-ROS-Bridge to record a map on a phone
2. Retrieve the map from Firebase
3. Use `as_graph` from `convert_json.py` to convert it to a graph, see `test_json.py`
4. Optionally save the new graph object as a pickle.

# Converting old-style graphs to python3 new style graphs
This is convoluted, since ROS depends on python2 and this project depends on python3.
Pickle has different encoding types between both python versions.

1. Convert the pickle to the new type using python2: `python2 convert_pickle.py src.pkl dest.pkl`.
   To find the  ROS datatypes, you need to `source /opt/ros/<installed_ros_distro>/setup.bash`.
2. Convert the python2 pickle to a python3 pickle: `python3 convert_pickle.py dest.pkl`.


## Dependencies
- [ARKit-ROS-Bridge](https://github.com/occamLab/ARKit-Ros-Bridge) to collect data
- [g2opy](https://github.com/uoip/g2opy) to work with the graphs.
  - There is an issue with newer versions of Eigen, a dependency of g2opy.
    [This pull request](https://github.com/uoip/g2opy/pull/16) fixes it.
  - If g2o is building for the wrong python version, see [this issue](https://github.com/uoip/g2opy/issues/9).

## Source Files
- `pose_graph.py`: This file describes the old type of posegraphs and is kept as a dependency for the converter.
- `graph.py` The new type of graph that uses the python g2o bindings.
- `convert_pickle.py`: This converts from the old to new type of posegraph.
- `convert_json.py`: There is new functionality to collect data straight from a phone and load a JSON file from FireBase.
  This contains functions to convert from json to the new graph type.
- `graph_utils.py`: Contains useful helper functions for graphs, such as converting them to a dict of arrays for plotting or integrating measurements into a path.
- `maximization_model.py`: Contains the maximization model to use for EM. See math.pdf for details.
- `plot_graph.py` Plot an input graph pickle.
  

## Test Files
- `check_errs.py`: Used to plot the difference between the g2o optimized vertex positions and the original vertex estimates.
- `get_subgraph_test.py`: Used to test subgraph extraction from `graph_utils.py`.
  This can be useful in assessing the jumpiness metric.
- `metrics.py`: Going to have to remind myself of this one.
