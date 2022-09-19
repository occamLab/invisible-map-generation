# Configs
The configs folder is intended to map certain parameters to other parameters in order to optimize workflows and minimize the length of optimization commands. Currently (9/18/2022), there are two configs in the config folder: firebase_device_config.json and ground_truth_mapping.json.

## firebase_device_config.json
The firebase_device_config.json intends to map specific firebase device ids with the name of the person associated with that id. This is to allow for downloading maps from firebase from a specific person's device, rather than wasting time waiting for the cache manager singleton to download every map from firebase.

Usage:
1. Add your name and device id to the firebase device config in the format "person_name": "firebase_device_id".
2. Add -fs "person_name" flag when calling optimize_graphs_and_manage_cache to download all of person_name's maps.

## ground_truth_mapping.json
The ground truth mapping json is intended to map gt datasets with corresponding maps. This allows for parameter sweeps and hollistic_optimize to be easily called with ground truth comparisons. The format goes as follows: "ground_truth_data_set_name" (no gt_ prefix)": ["list_of_map_names"]. This means that whenever an optimization is called with -g, the script looks in ground truth mapping to determine if there's a corresponding gt_data_set to compare the queried map to.