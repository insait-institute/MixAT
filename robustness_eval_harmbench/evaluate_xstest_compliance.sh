if [ $# -ne 1 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi


model_name=$1

./../robustness_eval_harmbench/eval_commands_compliance.sh $model_name