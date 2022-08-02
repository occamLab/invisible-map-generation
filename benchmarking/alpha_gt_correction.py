import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Map, SBA, Oblique, GT, Alpha
data_as_list = [[0, True, False, 0.4122251805184278, 0.4693017305863726],
                [0, False, False, 0.4693017305863726, 3.597106735269945],
                [1, True, True, 2.4830930692532625, 4.8034737353612975],
                [1, False, True, 1.0949620346049764, 10.84936043529909],
                [2, True, False, 0.351327362147805, 0.3695606772244322],
                [2, False, False, 0.38145472591695706, 3.4262794582283584],
                [3, True, False, 0.6071509617014544, 0.6077510086873326],
                [3, False, False, 0.7468231835444697, 0.7503525402429059],
                [4, True, False, 0.6341077701963096, 0.7729871372392112],
                [4, False, False, 0.8664202399348512, 2.5079394237087946],
                [5, True, True, 5.492737599382568, 28.09804851932197],
                [5, False, True, 1.5685393088650483, 8.750724311547673],
                [6, True, True, 2.9659351272042755, 3.726751634303463],
                [6, False, True, 2.7621308965933076, 3.3033897296799384],
                [7, True, True, 0.6125945027105751, 0.8383744614481091],
                [7, False, True, 0.8564900733897481, 3.211554203853354],
                [8, True, True, 3.8112978959625674, 3.8648293867972283],
                [8, False, True, 1.0060093991233812, 4.536009530216895]]
df = pd.DataFrame(data_as_list, columns=["MapID", "SBA", "Oblique", "GT", "Alpha"])

# Compare SBA to no SBA for individual metrics
# gt_sba_to_no_sba = stats.pearsonr(METRICS_DICT["sba"][0], METRICS_DICT["no_sba"][0])
# alpha_sba_to_no_sba = stats.pearsonr(METRICS_DICT["sba"][1], METRICS_DICT["no_sba"][1])
# No sba and sba are positively correlated, so if one is high the other will relatively be too. This shows that neither
# can save a particularly bad dataset.

# Compare SBA straight to SBA not straight
df_SBA_straight = df.loc[(df.Oblique == False) & (df.SBA == True), ['GT', 'Alpha']]
df_SBA_oblique = df.loc[(df.Oblique == True) & (df.SBA == True), ['GT', 'Alpha']]
print(df_SBA_oblique["GT"])


def plot_compare(df):
    """

    """
    x = np.arange(len(df_SBA_oblique))
    fig, ax = plt.subplots()
    print(df_SBA_oblique['GT'])
    rects1 = ax.bar(x - (0.35 / 2), df_SBA_oblique['GT'], 0.35, label="SBA GT")
    rects2 = ax.bar(x + (0.35 / 2), df_SBA_oblique['Alpha'], 0.35, label="SBA Alpha")

    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Metrics per Dataset")
    # ax.set_xticks(x, ["SBA GT", "SBA Alpha"])
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.show()


plot_compare(df_SBA_straight)

# gt_straight_sba_to_gt_ob_sba = stats.pearsonr(STRAIGHT_METRICS_DICT["sba"][0], OBLIQUE_METRICS_DICT["sba"][0])
# alpha_straight_sba_to_alpha_ob_sba = stats.pearsonr(STRAIGHT_METRICS_DICT["sba"][1], OBLIQUE_METRICS_DICT["sba"][1])

# print(f"gt sba to no sba: {gt_sba_to_no_sba} \n alpha sba to no sba: {alpha_sba_to_no_sba} \n ")


