if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_path> model_id"
    exit 1
fi



export HF_HOME=<path_to_your_huggingface_cache>
export HF_TOKEN=<your_huggingface_token>
export OPENAI_API_KEY=<your_openai_api_key>
./../../../utility_eval/mt_bench/evaluation_command.sh $1 $2 True