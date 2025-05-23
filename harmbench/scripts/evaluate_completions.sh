#!/bin/bash

source /opt/rh/devtoolset-10/enable

cls_path="$1"
behaviors_path=$2
completions_path=$3
save_path=$4
lora_exp="$5"
multiple_exp="$6"
begin_num="${7}"
end_num="${8}"
step_num="${9}"
final_num="${10}"
list_num="${11}"
temp_repeats="${12}"
window_book="${13}"
overlap_book="${14}"
window_lyrics="${15}"
overlap_lyrics="${16}"
threshold="${17}"

echo "cls_path=$cls_path"
echo "behaviors_path=$behaviors_path"
echo "completions_path=$completions_path"
echo "save_path=$save_path"
echo "lora_exp=$lora_exp"
echo "multiple_exp=$multiple_exp"
echo "begin_num=$begin_num"
echo "end_num=$end_num"
echo "step_num=$step_num"
echo "final_num=$final_num"
echo "list_num=$list_num"
echo "temp_repeats=$temp_repeats"
echo "window_book=$window_book"
echo "overlap_book=$overlap_book"
echo "window_lyrics=$window_lyrics"
echo "overlap_lyrics=$overlap_lyrics"
echo "threshold=$threshold"

cmd="python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path "

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

if [[ ! -z "$temp_repeats" ]]; then
    cmd="$cmd --temperature_repeats $temp_repeats"
fi

if [[ ! -z "$list_num" ]]; then
    cmd="$cmd --list_num \"$list_num\""
fi

if [[ ! -z "$window_book" ]]; then
    cmd="$cmd --window_size_book $window_book"
fi

if [[ ! -z "$window_lyrics" ]]; then
    cmd="$cmd --window_size_lyrics $window_lyrics"
fi

if [[ ! -z "$overlap_book" ]]; then
    cmd="$cmd --overlap_size_book $overlap_book"
fi

if [[ ! -z "$overlap_lyrics" ]]; then
    cmd="$cmd --overlap_size_lyrics $overlap_lyrics"
fi

if [[ ! -z "$threshold" ]]; then
    cmd="$cmd --threshold $threshold"
fi

cmd="$cmd --include_advbench_metric"
# Execute the command
eval $cmd
