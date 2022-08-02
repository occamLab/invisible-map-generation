import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def data_from_list(data, txt, sba, oblique):
    """
    This function takes in configured data and appends data read from the file specified int he arguments to it.

    Args:
        data: A list of already configured data to append new data to
        txt: A string representing the name of the file to get data from
        sba: A boolean representing whether this file contains SBA data or not
        oblique: A boolean representing whether the data in the file was recorded obliquely or not

    Returns:
    A list of lists where each line contains the configured data pulled from the file. The form of this data is:
    [Name of the Recording, SBA, min GT when GT is used, min GT when alpha is used, Oblique]
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


def plot_compare(df_this, graph_title):
    """
    This function plots the min GT when GT and Alpha are used to determine for the Pandas DataFrame provided.

    Args:
        df_this: A Pandas DataFrame to be plotted for comparing between GT and Alpha
        graph_title: A string representing the title of the plot

    """
    x = np.arange(len(df_this))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (0.35 / 2), df_this['GT'], 0.35, label="GT")
    rects2 = ax.bar(x + (0.35 / 2), df_this['Alpha'], 0.35, label="Alpha")

    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Metrics per Dataset")
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.title(graph_title)
    plt.show()


oblique = ["floor_2_obright", "floor_2_obleft", "floor_2_right_once", "209_occam_obleft_once", "mac-1-2-official"]

# Map Name, SBA, GT, Alpha, Oblique
data_mid = data_from_list([], "twenty-eighth-no-sba.txt", False, oblique)
data_as_list = data_from_list(data_mid, "twenty-eighth-sba.txt", True, oblique)
df = pd.DataFrame(data_as_list, columns=["MapName", "SBA", "GT", "Alpha", "Oblique"])

# Compare SBA straight to SBA not straight
df_SBA_straight = df.loc[(df.Oblique == False) & (df.SBA == True), ['GT', 'Alpha']]
df_SBA_oblique = df.loc[(df.Oblique == True) & (df.SBA == True), ['GT', 'Alpha']]

plot_compare(df_SBA_straight, "SBA Straight")
plot_compare(df_SBA_oblique, "SBA Oblique")

# Compare no SBA straight to no SBA not straight
df_no_SBA_straight = df.loc[(df.Oblique == False) & (df.SBA == False), ['GT', 'Alpha']]
df_no_SBA_oblique = df.loc[(df.Oblique == True) & (df.SBA == False), ['GT', 'Alpha']]

plot_compare(df_no_SBA_straight, "No SBA Straight")
plot_compare(df_no_SBA_oblique, "No SBA Oblique")



