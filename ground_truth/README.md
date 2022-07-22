# Ground Truth Data Generation

This folder is designed to create the ground truth data necessary to accurately evaluate the quality of IM's optimizations.

Ground truth = the actual location of the tag. Used to benchmark the current invisible maps system, since we can't tell how good the map is without knowing where the tags *should* be.

## Making a new ground truth dataset
We're using [rtabmap](https://github.com/krish-suresh/rtabmap) with a few modifications to spit out a .txt file of accurate tag data. We had to convert the data from quaternions to roll pitch yaw because we saw that the quaternions were incorrect for some reason.

Anyways, if you want to make another ground truth dataset, go to [this repo](https://github.com/occamLab/rtabmap) and read the README for instructions on how to do that. 

The instructions show allow you to produce a ".g2o" file containing translation and roll-pitch-yaw data. If you rename it to a .txt and add it to the gt_analysis_config (specifying a "gt_..." name to save it to cache), you  you can run `gt_data_generator.py` in this folder with the correct configuration specified in `gt_analysis_config.json` to produce a ground truth dataset that can then be integrated into IM's existing workflow. Note the command line argument requires you specify a test name!

The "PROCESSED_GRAPH_DATA_PATH" is only necessary if you want to do a visualization using the gt data generator. 