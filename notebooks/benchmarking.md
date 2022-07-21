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
Rotation metric: [0.02088048 3.89038216 0.02604206]
Maximum rotation: [ 0.22819989 11.8977332   0.33018249] (tag id: 324.0)

Pre-optimization value: 2.242
Minimum ground truth value: 1.529 (delta is -0.712)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 2.115 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 3.2e+10
>       alpha_all_before: 1.6e+10
> se3_not_gravity_before: 3.2e+10
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 0.065
>        alpha_all_after: 9.7e+02
>  se3_not_gravity_after: 0.065
>           psi2uv_after: 0
>          gravity_after: 0.00015

Parameters:
{
  "lin_vel_var": 1.0,
  "ang_vel_var": 1.0,
  "tag_sba_var": 1e-20
}
Maximum ground truth metric: 3.6695106632314607 (tag id: 301.0)
Ground Truth per Tag: 
 {300.0: 3.5801318142466045, 301.0: 3.6695106632314607, 302.0: 1.5473798941857566, 303.0: 2.7681401456275956, 304.0: 1.816855339538218, 305.0: 1.061351851892061, 306.0: 2.3700160764840006, 307.0: 0.7743125267970973, 308.0: 0.6801760472480655, 309.0: 1.8606485650211033, 310.0: 1.0647995685570992, 311.0: 1.0742994379015938, 312.0: 1.8762207604074057, 313.0: 0.8998147274546052, 314.0: 0.8799302051194969, 315.0: 1.1445353059420595, 316.0: 1.24852963206368, 317.0: 0.7834493569587327, 318.0: 1.4548555804223193, 319.0: 1.3115108902243486, 320.0: 1.1737874196515024, 321.0: 1.6077715984502425, 322.0: 1.240947804903118, 323.0: 1.4474630169135452, 324.0: 0.8965936895100026}

(Even more extreme parameters)

Rotation metric: [0.03741137 0.00495409 0.01081784]
Maximum rotation: [0.17787561 0.02539721 0.05164858] (tag id: 303.0)

Pre-optimization value: 2.242
Minimum ground truth value: 2.241 (delta is -0.001)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 3.814 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 2.4e+13
>       alpha_all_before: 1.2e+13
> se3_not_gravity_before: 2.4e+13
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 5.9e+09
>        alpha_all_after: 3e+09
>  se3_not_gravity_after: 5.9e+09
>           psi2uv_after: 0
>          gravity_after: 0.094

Parameters:
{
  "lin_vel_var": 2.1544346900318868e-11,
  "ang_vel_var": 1668.100537200059,
  "tag_sba_var": 1e-50
}
Maximum ground truth metric: 5.487071510533375 (tag id: 324.0)
Ground Truth per Tag: 
 {300.0: 3.386194754556807, 301.0: 3.168294411004407, 302.0: 1.0609905968160402, 303.0: 2.557945493660028, 304.0: 1.0280120722206783, 305.0: 0.8602978673515744, 306.0: 3.8007743122390503, 307.0: 1.5370311004505282, 308.0: 1.8863012910113872, 309.0: 2.4214654352068696, 310.0: 1.7741136369230879, 311.0: 1.1413789696175365, 312.0: 1.300061563554203, 313.0: 2.462164884753693, 314.0: 2.092198374457659, 315.0: 2.288883001985822, 316.0: 1.5927893584450181, 317.0: 1.54636796899783, 318.0: 1.3492038328361102, 319.0: 2.7503051677899926, 320.0: 2.5899075394266804, 321.0: 2.4792067498979176, 322.0: 2.078510462673607, 323.0: 3.3829726915192424, 324.0: 5.487071510533375}

7. "obleft" (fixed vertices, sba, og params, no -t)
Rotation metric: [114.23315454  27.7660842   51.20979159]
Maximum rotation: [179.63906631  70.63782039 179.29344399] (tag id: 319.0)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 69.266 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 1.1e+12
>       alpha_all_before: 5.6e+11
> se3_not_gravity_before: 1.1e+12
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 1.1e+12
>        alpha_all_after: 5.5e+11
>  se3_not_gravity_after: 1.1e+12
>           psi2uv_after: 0
>          gravity_after: 5.7e-13

Parameters:
{
  "lin_vel_var": 1e-10,
  "ang_vel_var": 1e-10,
  "tag_sba_var": 1e-10
}
Rotation metric: [114.23315454  27.7660842   51.20979159]
Maximum rotation: [179.63906631  70.63782039 179.29344399] (tag id: 319.0)
Maximum ground truth metric: 61.249634678639744 (tag id: 302.0)
Ground Truth per Tag: 
 {300.0: 45.37727291583269, 301.0: 50.33125394505156, 302.0: 61.249634678639744, 303.0: 35.642656823965055, 304.0: 38.275058524004535, 305.0: 35.87877222422735, 306.0: 44.54076776965328, 307.0: 38.01157974969274, 308.0: 45.55807981659313, 309.0: 28.66445843578908, 310.0: 44.76541485412417, 311.0: 48.754191805242115, 312.0: 47.270672140433135, 313.0: 49.708806115025034, 314.0: 49.160809107969754, 315.0: 46.54230293772589, 316.0: 18.920679194024213, 317.0: 53.97743040059188, 318.0: 43.60903493740817, 319.0: 57.41402340856481, 320.0: 10.965064435228676, 321.0: 41.17343727982026, 322.0: 16.260110591174715, 323.0: 9.185239061576455, 324.0: 21.651840099909954}

8. "obleft" (sba, fixed vertices (Tagpoint) no -t)
Rotation metric: [0.05945572 4.37268617 0.09247306]
Maximum rotation: [ 0.34007759 12.32630522  0.49722263] (tag id: 324.0)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 2.187 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 3.2e+10
>       alpha_all_before: 1.6e+10
> se3_not_gravity_before: 3.2e+10
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
----------------------------------
>         chi2_all_after: 0.5
>        alpha_all_after: -6.3e+03
>  se3_not_gravity_after: 0.5
>           psi2uv_after: 0
>          gravity_after: 0.0006

Parameters:
{
  "lin_vel_var": 0.5994842503189421,
  "ang_vel_var": 0.0359381366380464,
  "tag_sba_var": 1e-10
}
Rotation metric: [0.05945572 4.37268617 0.09247306]
Maximum rotation: [ 0.34007759 12.32630522  0.49722263] (tag id: 324.0)
Maximum ground truth metric: 3.7433895722760475 (tag id: 300.0)
Ground Truth per Tag: 
 {300.0: 3.7433895722760475, 301.0: 3.6905397227986647, 302.0: 1.5217051933735497, 303.0: 3.001622624284236, 304.0: 1.8300416135898478, 305.0: 1.1086318726012983, 306.0: 2.075946819490124, 307.0: 0.9462655765700195, 308.0: 0.7020567858214172, 309.0: 1.8479753253210018, 310.0: 1.1587374408188729, 311.0: 1.168132330030588, 312.0: 2.06081564798617, 313.0: 0.9587364265397035, 314.0: 0.8711145473065244, 315.0: 1.130268952125072, 316.0: 1.3225884979037397, 317.0: 0.7964929292157241, 318.0: 1.5094620943064196, 319.0: 1.3940523690412407, 320.0: 1.4797665903888073, 321.0: 1.4505536296624815, 322.0: 1.196963552851849, 323.0: 1.3782862042845603, 324.0: 0.8693364030382469}

9. "obleft" (sba, fixed TAG, -t)
Rotation metric: [6.46238182e-22 0.00000000e+00 4.50516944e-22]
Maximum rotation: [2.02358509e-21 0.00000000e+00 3.75545731e-21] (tag id: 304.0)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 3.819 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 1.1e+12
>       alpha_all_before: 5.6e+11
> se3_not_gravity_before: 1.1e+12
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
---------------------------------
>         chi2_all_after: 2.5e+10
>        alpha_all_after: 1.3e+10
>  se3_not_gravity_after: 2.5e+10
>           psi2uv_after: 0
>          gravity_after: 0.16

Parameters:
{
  "lin_vel_var": 1e-10,
  "ang_vel_var": 1e-10,
  "tag_sba_var": 1e-10
}
Rotation metric: [6.46238182e-22 0.00000000e+00 4.50516944e-22]
Maximum rotation: [2.02358509e-21 0.00000000e+00 3.75545731e-21] (tag id: 304.0)
Maximum ground truth metric: 5.496384793035149 (tag id: 324.0)
Ground Truth per Tag: 
 {300.0: 3.386990712119941, 301.0: 3.176143084186295, 302.0: 1.0428239370070729, 303.0: 2.5767414147650016, 304.0: 1.0304423145860009, 305.0: 0.8631450138711995, 306.0: 3.7917279759138602, 307.0: 1.5409647624531615, 308.0: 1.8831209864711065, 309.0: 2.4253600681574183, 310.0: 1.7752123049688935, 311.0: 1.1265191752411736, 312.0: 1.2807441222394507, 313.0: 2.4552563799249403, 314.0: 2.0940233497233405, 315.0: 2.2958465865270417, 316.0: 1.598862700654169, 317.0: 1.5494898923389755, 318.0: 1.3433394479857463, 319.0: 2.7556072032402694, 320.0: 2.5984500972347986, 321.0: 2.485152906817525, 322.0: 2.080946237050329, 323.0: 3.3900138230914716, 324.0: 5.496384793035149}

10. "obleft" (sba, fix waypoint, -t)
Rotation metric: [0.05945572 4.37268617 0.09247306]
Maximum rotation: [ 0.34007759 12.32630522  0.49722263] (tag id: 324.0)
Maximum difference metric (pre-optimized): 3.819 (tag id: 300.0)
Maximum difference metric (optimized): 2.187 (tag id: 300.0)
Fitness metrics: 
>        chi2_all_before: 3.2e+10
>       alpha_all_before: 1.6e+10
> se3_not_gravity_before: 3.2e+10
>          psi2uv_before: 0
>         gravity_before: 5.7e-13
----------------------------------
>         chi2_all_after: 0.5
>        alpha_all_after: -6.3e+03
>  se3_not_gravity_after: 0.5
>           psi2uv_after: 0
>          gravity_after: 0.0006

Parameters:
{
  "lin_vel_var": 0.5994842503189421,
  "ang_vel_var": 0.0359381366380464,
  "tag_sba_var": 1e-10
}
Rotation metric: [0.05945572 4.37268617 0.09247306]
Maximum rotation: [ 0.34007759 12.32630522  0.49722263] (tag id: 324.0)
Maximum ground truth metric: 3.7433895722760475 (tag id: 300.0)
Ground Truth per Tag: 
 {300.0: 3.7433895722760475, 301.0: 3.6905397227986647, 302.0: 1.5217051933735497, 303.0: 3.001622624284236, 304.0: 1.8300416135898478, 305.0: 1.1086318726012983, 306.0: 2.075946819490124, 307.0: 0.9462655765700195, 308.0: 0.7020567858214172, 309.0: 1.8479753253210018, 310.0: 1.1587374408188729, 311.0: 1.168132330030588, 312.0: 2.06081564798617, 313.0: 0.9587364265397035, 314.0: 0.8711145473065244, 315.0: 1.130268952125072, 316.0: 1.3225884979037397, 317.0: 0.7964929292157241, 318.0: 1.5094620943064196, 319.0: 1.3940523690412407, 320.0: 1.4797665903888073, 321.0: 1.4505536296624815, 322.0: 1.196963552851849, 323.0: 1.3782862042845603, 324.0: 0.8693364030382469}
