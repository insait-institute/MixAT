#!/bin/bash
source /opt/rh/devtoolset-10/enable

method_name="$1"
experiment_name="$2"
behaviors_path="$3"
save_dir="$4"
start_idx="$5"
end_idx="$6"
behavior_ids_subset="$7"
lora_exp="$8"
multiple_exp="$9"
begin_num="${10}"
end_num="${11}"
step_num="${12}"
final_num="${13}"
list_num="${14}"
run_id="${15}"
overwrite="${16}"
verbose="${17}"

echo "method_name=$method_name"
echo "experiment_name=$experiment_name"
echo "behaviors_path=$behaviors_path"
echo "save_dir=$save_dir"
echo "start_idx=$start_idx"
echo "end_idx=$end_idx"
echo "beahavior_ids_subset=$behavior_ids_subset"
echo "lora_exp=$lora_exp"
echo "multiple_exp=$multiple_exp"
echo "begin_num=$begin_num"
echo "end_num=$end_num"
echo "step_num=$step_num"
echo "final_num=$final_num"
echo "list_num=$list_num"
echo "run_id=$run_id"
echo "overwrite=$overwrite"
echo "verbose=$verbose"

# Construct the base command
cmd="python -u generate_test_cases.py \
    --method_name $method_name \
    --experiment_name $experiment_name \
    --behaviors_path $behaviors_path \
    --save_dir $save_dir"

# Check if start_idx is provided and not empty
if [ ! -z "$start_idx" ]; then
    cmd="$cmd --behavior_start_idx $start_idx"
fi
# Check if end_idx is provided and not empty
if [ ! -z "$end_idx" ]; then
    cmd="$cmd --behavior_end_idx $end_idx"
fi
# Check if lora_exp is provided and True
if [[ ! -z "$lora_exp" && "$lora_exp" == "True" ]]; then
    cmd="$cmd --lora_scaling_experiments"
fi

if [[ ! -z "$multiple_exp" && "$multiple_exp" == "True" ]]; then
    cmd="$cmd --multiple_checkpoints"
fi

if [[ ! -z "$begin_num" ]]; then
    cmd="$cmd --begin $begin_num"
fi

if [[ ! -z "$end_num" ]]; then
    cmd="$cmd --end $end_num"
fi

if [[ ! -z "$step_num" ]]; then
    cmd="$cmd --step $step_num"
fi

if [[ ! -z "$final_num" ]]; then
    cmd="$cmd --final $final_num"
fi

if [[ ! -z "$list_num" ]]; then
    cmd="$cmd --list_num $list_num"
fi
# Check if behavior_ids_subset is provided and not empty
if [ ! -z "$behavior_ids_subset" ]; then
    cmd="$cmd --behavior_ids_subset $behavior_ids_subset"
fi
# Check if run_id is provided and not empty
if [ ! -z "$run_id" ]; then
    cmd="$cmd --run_id $run_id"
fi
# Check if overwrite is provided and equals "True"
if [[ ! -z "$overwrite" && "$overwrite" == "True" ]]; then
    cmd="$cmd --overwrite"
fi
# Check if verbose is provided and equals "True"
if [[ ! -z "$verbose" && "$verbose" == "True" ]]; then
    cmd="$cmd --verbose"
fi



# Execute the command
echo "Eval started"
eval $cmd
