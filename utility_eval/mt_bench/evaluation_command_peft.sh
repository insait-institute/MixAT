#!/bin/bash


if [ $# -ne 4 ]; then
    echo "Usage: $0 <model_path> model_id peft_path load_in_4bit"
    exit 1
fi

if [ "$4" == "True" ]; then
  python gen_model_answer.py --model-path $1 --model-id $2 --peft $3 --load_in_4bit --dtype bfloat16
else
  python gen_model_answer.py --model-path $1 --model-id $2 --peft $3 --dtype bfloat16
fi

python gen_judgment.py --model-list $2  --parallel 8
python show_result.py --model-list $2 