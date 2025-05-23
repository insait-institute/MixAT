if [ $# -ne 3 ]; then
    echo "Usage: $0  <model> <peft>(None if no) <model_name>"
    exit 1
fi

export HF_HOME=<your_huggingface_cache_path>
export HF_TOKEN=<your_huggingface_token>
python evaluation.py $1 $2 $3 None True False

