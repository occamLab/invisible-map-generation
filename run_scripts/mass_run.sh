# The purpose of this file is to run multiple parameter sweeps in succession,
# while logging the output to a specific txt file.

# Usage:
#     1. Change the listOfFilesToRun list to be a list of maps that you wish to be optimized
#     2. Change log_file_name to the desired output file to store the sweep results
#     3. Change line 14 to include all the optimization flags (ex. SBA or no SBA) that are desired
#     4. Run the shell script from the root direction of the repository

declare -a file_names=("*desnat_straight_once*" "*floor_2_obright*" "*desnat_straight_three*" "*209-to-occam*" "*robolab_straight_on*" "*floor_2_obleft*" "*floor_2_right_once*" "straight_209_occam" "*209_occam_obleft_once*" "*mac-1-2-official*")
log_file_name=sweep_test.txt
for file_name in ${file_names[@]}; do
    echo ${file_name} >> $log_file_name
    python3 run_scripts/optimize_graphs_and_manage_cache.py -g --pso 1 -s -p ${file_name} --np 8 >> $log_file_name
    echo >> $log_file_name
done
