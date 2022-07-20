declare -a listOfFilesToRun=("*floor_2_obright*" "*floor_2_obleft*" "*floor_2_right_once*" "straight_209_occam" "*209_occam_obleft_once*" "*mac-1-2-official*")
# echo "*floor_2 *" >> nineteenth.txt
# python3 run_scripts/optimize_graphs_and_manage_cache.py -g -nsb --pso 0 -s -p "floor_2 *" --np 8 >> nineteenth.txt
# echo >> nineteenth.txt
for fileName in ${listOfFilesToRun[@]}; do
    echo ${fileName} >> nineteenth.txt
    python3 run_scripts/optimize_graphs_and_manage_cache.py -g -nsb --pso 0 -s -p ${fileName} --np 8 >> nineteenth.txt
    echo >> nineteenth.txt
done
