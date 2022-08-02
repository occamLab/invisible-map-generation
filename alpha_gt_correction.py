import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def data_from_list(data, txt, sba):
    """

    """
    # Convert txt to list by lines
    with open(txt) as f:
        lines_list = [line.rstrip() for line in f.readlines()]

    # Extract data from the main list
    data_final = data
    dataset_names = [name for name in lines_list if '*' in name]

    for dataset in dataset_names:
        # Reset list to start from current index
        name_index = lines_list.index(dataset)
        lines_list = lines_list[name_index:]

        # Get alpha based gt for dataset
        line_before_alpha_index = lines_list.index("For map based on min alpha,")
        alpha_pre = lines_list[line_before_alpha_index + 1]
        alpha = float(alpha_pre[9:alpha_pre.index("(")])
        print(alpha)

        # Get gt based gt for dataset
        line_before_gt_index = lines_list.index("For map based on min gt,")
        gt_pre = lines_list[line_before_gt_index + 1]
        gt = float(gt_pre[9:gt_pre.index("(")])

        data_final.append([dataset, sba, gt, alpha])
    return data


data_mid = data_from_list([], "twenty-eighth-no-sba.txt", False)
data_final = data_from_list(data_mid, "twenty-eighth-sba.txt", True)

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


def plot_compare(df_this):
    """

    """
    x = np.arange(len(df_this))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (0.35 / 2), df_this['GT'], 0.35, label="SBA GT")
    rects2 = ax.bar(x + (0.35 / 2), df_this['Alpha'], 0.35, label="SBA Alpha")

    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Metrics per Dataset")
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.show()


# plot_compare(df_SBA_straight)



