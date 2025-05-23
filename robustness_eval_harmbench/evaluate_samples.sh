if [ $# -ne 2 ]; then
    echo "Usage: $0 <attack> <model>"
    exit 1
fi

export HF_HOME=<your_huggingface_cache_directory>
export HF_TOKEN=<your_huggingface_token>
./../robustness_eval_harmbench/evaluation_command.sh $1 $2