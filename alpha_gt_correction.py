import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

oblique = ["floor_2_obright", "floor_2_obleft", "floor_2_right_once", "209_occam_obleft_once", "mac-1-2-official"]

def data_from_list(data, txt, sba, oblique):
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

        # Get gt based gt for dataset
        line_before_gt_index = lines_list.index("For map based on min gt,")
        gt_pre = lines_list[line_before_gt_index + 1]
        gt = float(gt_pre[9:gt_pre.index("(")])

        # Add variable telling if data is obliquely recorded or not
        if dataset[1:-1] in oblique:
            oblique_bool = True
        else:
            oblique_bool = False
        data_final.append([dataset[1:-1], sba, gt, alpha, oblique_bool])
    return data

# Map Name, SBA, GT, Alpha, Oblique
data_mid = data_from_list([], "twenty-eighth-no-sba.txt", False, oblique)
data_as_list = data_from_list(data_mid, "twenty-eighth-sba.txt", True, oblique)

df = pd.DataFrame(data_as_list, columns=["MapName", "SBA", "GT", "Alpha", "Oblique"])

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


plot_compare(df_SBA_straight)



