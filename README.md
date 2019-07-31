# Invisible Map Generation

This repository is a refactor and extension of the work done in [occamlab/assistive_apps](https://github.com/occamLab/assistive_apps/tree/summer2018) to generate maps.

## Creating a map
1. Collect data by running the data_collection script in [occamlab/assistive_apps](https://github.com/occamLab/assistive_apps/tree/summer2018)'s navigation_prototypes project and name the output file 'academic_center.pkl'

2. Get an initial optimized graph by running this repository's 'optimization.py' file.

3. Iterate on the optimization by running 'glue.py' repeatedly.

## Dependencies
- [ARKit-ROS-Bridge](https://github.com/occamLab/ARKit-Ros-Bridge) to collect data
- [g2opy](https://github.com/uoip/g2opy) to work with the graphs.
  - There is an issue with newer versions of Eigen, a dependency of g2opy.
    [This pull request](https://github.com/uoip/g2opy/pull/16) fixes it.
  - If g2o is building for the wrong python version, see [this issue](https://github.com/uoip/g2opy/issues/9)
