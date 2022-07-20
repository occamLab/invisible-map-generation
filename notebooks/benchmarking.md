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

________________________________________________________________________________________________________________________

1. "obleft" (no tags removed, no pixel changed)

Rotation metric: [2.93711052 8.91921601 2.23330769]
Maximum rotation: [22.65248635 48.304836   11.11167463] (tag id: 313.0)

Pre-optimization value: 2.242
Minimum ground truth value: 6.382 (delta is 4.141)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 11.944 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 7.8e+16
>       alpha_all_before: 3.9e+16
> se3_not_gravity_before: 5.6e-13
>          psi2uv_before: 7.8e+16
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 3.7e+14
>        alpha_all_after: 1.8e+14
>  se3_not_gravity_after: 2.2e+14
>           psi2uv_after: 1.5e+14
>          gravity_after: 31

Parameters:
{
  "lin_vel_var": 4.641588833612782e-07,
  "ang_vel_var": 1e-10,
  "tag_sba_var": 2.782559402207126e-08
}

Maximum ground truth metric: 24.657449359152356 (tag id: 313.0)
Ground Truth per Tag: 
 {300.0: 3.142539492806661, 301.0: 5.878776197102793, 302.0: 1.448725800629373, 303.0: 3.2743382752759373, 304.0: 1.4194270506806397, 305.0: 2.976084167509242, 306.0: 11.282379803892882, 307.0: 6.504619804085849, 308.0: 2.416274458098255, 309.0: 8.1058535122978, 310.0: 8.56023299632019, 311.0: 4.070184937603785, 312.0: 10.593192313882657, 313.0: 24.657449359152356, 314.0: 2.3863502332912403, 315.0: 8.524760312771832, 316.0: 17.830692241390345, 317.0: 1.766600340290413, 318.0: 4.167676289192076, 319.0: 8.468913452382685, 320.0: 4.342575608645809, 321.0: 2.782668170189564, 322.0: 5.238019316257124, 323.0: 4.755123924916405, 324.0: 4.965687933707944}

2. "obleft" (tags removed, no pixels changed)
Rotation metric: [2.93711052 8.91921601 2.23330769]
Maximum rotation: [22.65248635 48.304836   11.11167463] (tag id: 313.0)

Pre-optimization value: 2.242
Minimum ground truth value: 6.382 (delta is 4.141)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 11.944 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 7.8e+16
>       alpha_all_before: 3.9e+16
> se3_not_gravity_before: 5.6e-13
>          psi2uv_before: 7.8e+16
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 3.7e+14
>        alpha_all_after: 1.8e+14
>  se3_not_gravity_after: 2.2e+14
>           psi2uv_after: 1.5e+14
>          gravity_after: 31

Parameters:
{
  "lin_vel_var": 4.641588833612782e-07,
  "ang_vel_var": 1e-10,
  "tag_sba_var": 2.782559402207126e-08
}

Maximum ground truth metric: 24.657449359152356 (tag id: 313.0)
Ground Truth per Tag: 
 {300.0: 3.142539492806661, 301.0: 5.878776197102793, 302.0: 1.448725800629373, 303.0: 3.2743382752759373, 304.0: 1.4194270506806397, 305.0: 2.976084167509242, 306.0: 11.282379803892882, 307.0: 6.504619804085849, 308.0: 2.416274458098255, 309.0: 8.1058535122978, 310.0: 8.56023299632019, 311.0: 4.070184937603785, 312.0: 10.593192313882657, 313.0: 24.657449359152356, 314.0: 2.3863502332912403, 315.0: 8.524760312771832, 316.0: 17.830692241390345, 317.0: 1.766600340290413, 318.0: 4.167676289192076, 319.0: 8.468913452382685, 320.0: 4.342575608645809, 321.0: 2.782668170189564, 322.0: 5.238019316257124, 323.0: 4.755123924916405, 324.0: 4.965687933707944}

3. "obleft" (tags removed, pixels changed)

Rotation metric: [2.59381105 7.44253078 2.96369978]
Maximum rotation: [18.13965817 41.2318685  18.39793197] (tag id: 316.0)

Pre-optimization value: 2.242
Minimum ground truth value: 5.761 (delta is 3.519)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 11.201 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 1.7e+08
>       alpha_all_before: 8.7e+07
> se3_not_gravity_before: 1.6e-21
>          psi2uv_before: 1.7e+08
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 7.2e+05
>        alpha_all_after: 3.5e+05
>  se3_not_gravity_after: 3.4e+05
>           psi2uv_after: 3.8e+05
>          gravity_after: 19

Parameters:
{
  "lin_vel_var": 0.0359381366380464,
  "ang_vel_var": 0.0359381366380464,
  "tag_sba_var": 10.0
}

Maximum ground truth metric: 21.481087953057422 (tag id: 313.0)
Ground Truth per Tag: 
 {300.0: 3.301719861203802, 301.0: 5.668474780022673, 302.0: 1.2209720298517537, 303.0: 3.0530086593338943, 304.0: 1.2067726764719253, 305.0: 2.1411446204116302, 306.0: 9.546652836356728, 307.0: 1.717507429494796, 308.0: 1.3034048164263543, 309.0: 8.971733830682783, 310.0: 2.111887692369366, 311.0: 3.1061832992395915, 312.0: 6.38807843308408, 313.0: 21.481087953057422, 314.0: 5.1328812290336305, 315.0: 7.11433054750406, 316.0: 16.89698728415909, 317.0: 5.123163093249329, 318.0: 5.736199725268335, 319.0: 8.060143921869942, 320.0: 5.618432968380393, 321.0: 2.916672131876939, 322.0: 6.017840465203823, 323.0: 5.308504697469891, 324.0: 4.869664809278985}

4. "obleft" (no throw out tags, no sparse bundle)

Rotation metric: [0.26374567 0.02981307 0.09793604]
Maximum rotation: [1.47343267 0.19299232 0.76119507] (tag id: 303.0)

Pre-optimization value: 2.242
Minimum ground truth value: 2.238 (delta is -0.004)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 3.812 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 1.1e+12
>       alpha_all_before: 5.6e+11
> se3_not_gravity_before: 1.1e+12
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 2.6e+09
>        alpha_all_after: 1.3e+09
>  se3_not_gravity_after: 2.6e+09
>           psi2uv_after: 0
>          gravity_after: 0.13

Parameters:
{
  "lin_vel_var": 1e-10,
  "ang_vel_var": 10.0,
  "tag_sba_var": 1e-10
}
Maximum ground truth metric: 5.469132222463389 (tag id: 324.0)
Ground Truth per Tag: 
 {300.0: 3.3682988829168674, 301.0: 3.1034900428644687, 302.0: 1.1238057019606378, 303.0: 2.4515862310354812, 304.0: 1.0226561410172064, 305.0: 0.822306100490717, 306.0: 3.815690845732027, 307.0: 1.515200490445289, 308.0: 1.8799756052059922, 309.0: 2.399905361214117, 310.0: 1.7945091780718514, 311.0: 1.2045736735894002, 312.0: 1.3591706531508572, 313.0: 2.454669647738528, 314.0: 2.0998532309050724, 315.0: 2.277809419831967, 316.0: 1.5873092743945434, 317.0: 1.5481110084270018, 318.0: 1.3900526458699252, 319.0: 2.7532729277830215, 320.0: 2.5752872946765093, 321.0: 2.4713489897561756, 322.0: 2.0847686720205347, 323.0: 3.37121406150421, 324.0: 5.469132222463389}

5. "obleft" (no sba, with changing pixel coordinates)

Rotation metric: [0.26374567 0.02981307 0.09793604]
Maximum rotation: [1.47343267 0.19299232 0.76119507] (tag id: 303.0)

Pre-optimization value: 2.242
Minimum ground truth value: 2.238 (delta is -0.004)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 3.812 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 1.1e+12
>       alpha_all_before: 5.6e+11
> se3_not_gravity_before: 1.1e+12
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 2.6e+09
>        alpha_all_after: 1.3e+09
>  se3_not_gravity_after: 2.6e+09
>           psi2uv_after: 0
>          gravity_after: 0.13

Parameters:
{
  "lin_vel_var": 1e-10,
  "ang_vel_var": 10.0,
  "tag_sba_var": 1e-10
}

*** Pixel coordinates changing doesn't do anything if no sba

6. "obleft" (no sba, lower parameters)
