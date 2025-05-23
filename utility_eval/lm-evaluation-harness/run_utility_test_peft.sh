if [ $# -ne 2 ]; then
    echo "Usage: $0 <model> <peft>"
    exit 1
fi


export HF_HOME=<path_to_your_huggingface_cache>
export HF_TOKEN=<your_huggingface_token>
lm_eval --model hf \
    --model_args pretrained=$1,parallelize=True,dtype=bfloat16,load_in_4bit=True,peft=$2 \
    --tasks arc_easy,arc_challenge,mmlu \
    --device cuda:0 \
    --batch_size auto:12 \
    --output_path ../utility_eval/results/{$1}_{$2} \
    --log_samples \
    --cache_requests true