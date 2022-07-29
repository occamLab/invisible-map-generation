declare -a listOfFilesToRun=("*desnat_straight_once*" "*floor_2_obright*" "*desnat_straight_three*" "*209-to-occam*" "*robolab_straight_on*" "*floor_2_obleft*" "*floor_2_right_once*" "straight_209_occam" "*209_occam_obleft_once*" "*mac-1-2-official*")
# echo "*floor_2 *" >> nineteenth.txt
# python3 run_scripts/optimize_graphs_and_manage_cache.py -g -nsb --pso 0 -s -p "floor_2 *" --np 8 >> nineteenth.txt
# echo >> nineteenth.txt
for fileName in ${listOfFilesToRun[@]}; do
    echo ${fileName} >> twenty-eighth-no-sba.txt
    python3 run_scripts/optimize_graphs_and_manage_cache.py -g --pso 1 -s -p ${fileName} --np 8 >> twenty-eighth-no-sba.txt
    echo >> twenty-eighth-no-sba.txt
    echo
    echo ${fileName} >> twenty-eighth-sba.txt
    python3 run_scripts/optimize_graphs_and_manage_cache.py -g -s -p ${fileName} --np 8 >> twenty-eighth-sba.txt
    echo >> twenty-eighth-sba.txt
    echo
done
