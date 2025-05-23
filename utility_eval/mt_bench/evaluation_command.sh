#!/bin/bash



if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_path> model_id load_in_4bit"
    exit 1
fi

if [ "$3" == "True" ]; then
  python gen_model_answer.py --model-path $1 --model-id $2 --load_in_4bit --dtype bfloat16
else
  python gen_model_answer.py --model-path $1 --model-id $2 --dtype bfloat16
fi
python gen_judgment.py --model-list $2 --parallel 8
python show_result.py --model-list $2