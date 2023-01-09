"""
Seeking to visualize the results of our alpha testing.

Will produce one graph per dataset ran, CTRL+W to exit out and see the next one.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WIDTH = 0.25    
DATA_PATH = "datasets/alpha_test_results.json"

def main():
    
        
    with open(DATA_PATH,"r") as data_file:
        alpha_data = json.load(data_file)

    datasets = alpha_data.keys()

    for dataset in datasets:
        pre_opt_gts = []
        min_gts = []
        alpha_min_gts = []
        
        results = alpha_data[dataset]
        tests = results.keys()
        x_axis = np.arange(len(tests))
        for test in tests:
            pre_opt_gts.append(results[test]["pre_opt_gt"])
            min_gts.append(results[test]["min_gt"])
            alpha_min_gts.append(results[test]["alpha_min_gt"])
        
        
        bar1 = plt.bar(x_axis, pre_opt_gts, WIDTH, color = 'r')
        bar2 = plt.bar(x_axis+WIDTH, min_gts, WIDTH, color='g')
        bar3 = plt.bar(x_axis+2*WIDTH, alpha_min_gts, WIDTH, color='b')
        
        plt.xlabel("tests")
        plt.ylabel("score")
        plt.title(f"{dataset} scores")
        
        x_labels = ["0","t5", "t10", "o1", "o2.5", "t5o1", "t10o2.5"]
        legend = ("pre_opt_gt", "min_gt", "alpha_min_gt")
        plt.xticks(x_axis+WIDTH,x_labels)
        plt.legend( (bar1, bar2, bar3), legend )   
        plt.savefig(f"{dataset} scores")
        plt.show()
   

if __name__ == "__main__":
    main()