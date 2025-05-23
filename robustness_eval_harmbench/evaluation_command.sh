base_save_dir="../robustness_eval_harmbench/results"
method_name=$1
experiment=$2
save_dir="$base_save_dir/$method_name/$experiment/test_cases"

experiment_dir="$base_save_dir/$method_name/$experiment"
model_name=$2 
test_cases_path="$experiment_dir/test_cases/test_cases.json"
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_test.csv"
max_new_tokens=512
incremental_update="False"
save_path_completions="$experiment_dir/completions/${model_name}.json"

cls_path="cais/HarmBench-Llama-2-13b-cls" 
completions_path="$base_save_dir/$method_name/$experiment/completions/$model_name.json" 
save_path="$base_save_dir/$method_name/$experiment/results/$model_name.json"


./scripts/merge_test_cases.sh $method_name $save_dir
./scripts/generate_completions.sh $model_name $behaviors_path $test_cases_path $save_path_completions $max_new_tokens $incremental_update
./scripts/evaluate_completions.sh $cls_path $behaviors_path $completions_path $save_path