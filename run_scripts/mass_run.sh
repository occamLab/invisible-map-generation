# The purpose of this file is to run multiple parameter sweeps in succession,
# while logging the output to a specific txt file.

# Usage:
#     1. Change the listOfFilesToRun list to be a list of maps that you wish to be optimized
#     2. Change log_file_name to the desired output file to store the sweep results
#     3. Change line 14 to include all the optimization flags (ex. SBA or no SBA) that are desired
#     4. Run the shell script from the root direction of the repository

declare -a file_names=("generated_23-01-03-23-46-16.json" "generated_23-01-03-23-48-03.json" "generated_23-01-03-23-49-26.json" "generated_23-01-03-23-50-57.json" "generated_23-01-03-23-52-28.json" "generated_23-01-03-23-53-51.json" "generated_23-01-03-23-55-12.json" "generated_23-01-03-23-46-59.json" "generated_23-01-03-23-48-32.json" "generated_23-01-03-23-49-40.json" "generated_23-01-03-23-51-16.json" "generated_23-01-03-23-52-37.json" "generated_23-01-03-23-54-02.json" "generated_23-01-03-23-55-21.json" "generated_23-01-03-23-47-25.json" "generated_23-01-03-23-48-58.json" "generated_23-01-03-23-49-54.json" "generated_23-01-03-23-51-31.json" "generated_23-01-03-23-52-47.json" "generated_23-01-03-23-54-21.json" "generated_23-01-03-23-55-46.json")
log_file_name=alpha_test_jan_third.txt
for file_name in ${file_names[@]}; do
    echo ${file_name} >> $log_file_name
    python3 run_scripts/optimize_graphs_and_manage_cache.py -g --pso 1 -s -p ${file_name} --np 8 >> $log_file_name
    echo >> $log_file_name
done
