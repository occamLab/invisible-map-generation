# Benchmarking the Invisible Maps Algorithm

We've implemented a few datasets to evaluate the effectiveness of the g2o algorithm behind invisible maps' optimization. 

## Definitions:

"Ground truth metric" is computed as follows:

Calculates the transforms from the anchor tag to each other tag for the optimized and the ground truth tags, then compares the transforms and finds the difference in the translation components.

## Workflow
There are a number of different "weights" that can be used that value certain elements of the map more highly when optimizing the map. There's additionally an option to parameter sweep, which (to us) is a bit of a black box that we're just assuming sweeps every relevant parameter(?). What we're looking at here is the ground truth metric at pre-optimization and comparing it to the ground truth metric the parameter sweep finds as optimal.

We currently have 3 different datasets that we're using to evaluate the effectiveness of the ground truth metric. The names here are referenced in the `rtabmap/gt_analysis_config.json` file.

1. "mac_full"

A bit of a misnomer. "mac_full" only maps out the 2nd and 3rd floors of the MAC.

WITH LIDAR IPAD (data is more accurate than it should be)

**pre-optimization ground truth**: 0.987

**parameter sweep**: 1.280 with parameters below

{
  "lin_vel_var": 0.0001668100537200059,
  "ang_vel_var": 0.01,
  "tag_sba_var": 0.21544346900318834
}

2. "desnat"

A short path from the endcap of the hallway to the desnat storage room

**pre-optimzation ground truth**: 0.486

**parameter sweep**: 0.458 with parameters below

{
  "lin_vel_var": 0.0012915496650148827,
  "ang_vel_var": 0.01,
  "tag_sba_var": 0.01
}

3. "occam"

A mapping of the tags in the OCCaM classroom.

**pre-optimzation ground truth**: 0.389

**parameter sweep**: 0. 373 with parameters below

{
  "lin_vel_var": 0.0001668100537200059,
  "ang_vel_var": 7.742636826811277e-06,
  "tag_sba_var": 0.046415888336127774
}