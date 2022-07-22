# Ground Truth Data Generation

This folder is designed to create the ground truth data necessary to accurately evaluate the quality of IM's optimizations.

Ground truth = the actual location of the tag. Used to benchmark the current invisible maps system, since we can't tell how good the map is without knowing where the tags *should* be.

We're using [rtabmap](https://github.com/krish-suresh/rtabmap) with a few modifications to spit out a .txt file of accurate tag data. We had to convert the data from quaternions to roll pitch yaw because we saw that the quaternions were incorrect for some reason.

Anyways, if you want to make another ground truth dataset, you just have to email [Ayush](mailto:achakraborty@olin.edu) and ask him for his version of the RTABmap code that produces the RPY datafile we need :).

Afterwards, you can run `gt_data_analysis.py` in this folder with the correct configuration specified in `gt_analysis_config.json` to produce a ground truth dataset that can then be integrated into IM's existing workflow. Note the command line argument requires you specify a test name!