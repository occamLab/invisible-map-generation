import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# GT values once map selected by GT and alpha respectively with first row GT and second row alpha
METRICS_DICT = {"sba": [[0.4122251805184278, 2.4830930692532625, 0.351327362147805, 0.6071509617014544,
                         0.6341077701963096, 5.492737599382568, 2.9659351272042755, 0.6125945027105751,
                         3.8112978959625674],
                        [0.47120212602422085, 4.8034737353612975, 0.3695606772244322, 0.6077510086873326,
                         0.7729871372392112, 28.09804851932197, 3.726751634303463, 0.8383744614481091,
                         3.8648293867972283]
                        ],
                "no_sba": [[0.4693017305863726, 1.0949620346049764, 0.38145472591695706, 0.7468231835444697,
                            0.8664202399348512, 1.5685393088650483, 2.7621308965933076, 0.8564900733897481,
                            1.0060093991233812],
                           [3.597106735269945, 10.84936043529909, 3.4262794582283584, 0.7503525402429059,
                            2.5079394237087946, 8.750724311547673, 3.3033897296799384, 3.211554203853354,
                            4.536009530216895]
                           ]}
INDEX_OBLIQUE = [1, 5, 6, 7, 8]
INDEX_STRAIGHT = [0, 2, 3, 4]
OBLIQUE_METRICS_DICT = {"sba": [[METRICS_DICT["sba"][0][i] for i in INDEX_OBLIQUE],
                                [METRICS_DICT["sba"][1][i] for i in INDEX_OBLIQUE]
                                ],
                        "no_sba": [[METRICS_DICT["no_sba"][0][i] for i in INDEX_OBLIQUE],
                                   [METRICS_DICT["no_sba"][0][i] for i in INDEX_OBLIQUE]
                                   ]}

STRAIGHT_METRICS_DICT = {"sba": [[METRICS_DICT["sba"][0][i] for i in INDEX_STRAIGHT],
                                 [METRICS_DICT["sba"][1][i] for i in INDEX_STRAIGHT]
                                 ],
                         "no_sba": [[METRICS_DICT["no_sba"][0][i] for i in INDEX_STRAIGHT],
                                    [METRICS_DICT["no_sba"][0][i] for i in INDEX_STRAIGHT]
                                    ]}


gt_sba_to_no_sba = stats.pearsonr(METRICS_DICT["sba"][0], METRICS_DICT["no_sba"][0])
alpha_sba_to_no_sba = stats.pearsonr(METRICS_DICT["sba"][1], METRICS_DICT["no_sba"][1])

# gt_straight_sba_to_gt_ob_sba = stats.pearsonr(STRAIGHT_METRICS_DICT["sba"][0], OBLIQUE_METRICS_DICT["sba"][0])
# alpha_straight_sba_to_alpha_ob_sba = stats.pearsonr(STRAIGHT_METRICS_DICT["sba"][1], OBLIQUE_METRICS_DICT["sba"][1])

print(f"gt sba to no sba: {gt_sba_to_no_sba} \n alpha sba to no sba: {alpha_sba_to_no_sba} \n ")


