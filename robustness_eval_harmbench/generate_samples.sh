if [ $# -ne 2 ]; then
    echo "Usage: $0 <attack> <model>"
    exit 1
fi


base_save_dir="../robustness_eval_harmbench/results"
method_name=$1
experiment_name=$2
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_test.csv"
save_dir="${base_save_dir}/${method_name}/${experiment_name}/test_cases"
start_idx=0
end_idx=40
behavior_ids_subset="0,1,2,3,5,6,7,8,10,12,13,14,16,18,19,21,22,23,24,25,28,29,30,31,33,34,35,37,39,40,41,42,43,44,45,46,47,49,50,51"
run_id=""
overwrite="False"
verbose="True"

export HF_HOME=<your_huggingface_cache_directory>
export HF_TOKEN=<your_huggingface_token>
./scripts/generate_test_cases.sh $method_name $experiment_name $behaviors_path $save_dir $start_idx $end_idx $behavior_ids_subset