#!/bin/bash
############# This script should not be run directly, it is called by evaluate_xstest_compliance.sh #############

base_save_dir="../robustness_eval_harmbench/results"
method_name='XSTest'
experiment=$1
save_dir="$base_save_dir/$method_name/$experiment/test_cases"
lora_exp="False"

experiment_dir="$base_save_dir/$method_name"
model_name=$1 # from configs/model_configs/models.yaml
behaviors_path="./data/xstest/xstest_v2_prompts.csv"
test_cases_path=$behaviors_path
max_new_tokens=512
incremental_update="False"
save_path_completions="$experiment_dir/completions/${model_name}.csv"

beg_num=15
end_num=750
step_num=75
final_num=760

num_list=$2
if [ -z "$num_list" ]; then
    multiple_args="False"
else
    multiple_args="True $beg_num $end_num $step_num $final_num $num_list"
fi

echo "running $method_name $experiment $num_list"

export HF_HOME=<your_huggingface_cache_directory>
export HF_TOKEN=<hf_token>
export OPENAI_API_KEY=<openai_token>

./scripts/generate_completions.sh $model_name $behaviors_path $test_cases_path $save_path_completions $max_new_tokens $incremental_update False $multiple_args
python -u ./classify_completions_xstest.py --input-path $save_path_completions --output-path $save_path_completions
