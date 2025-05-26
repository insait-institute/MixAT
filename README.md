# MixAT

This repository contains the code for the paper **MixAT: Combining Continuous and Discrete Adversarial Training Methods**.

## Trained Adapters
Trained adapters with the MixAT and MixAT+GCG methods can be found on [Hugging Face](https://huggingface.co/collections/INSAIT-Institute/mixat-68345b175d381301e7520e2a)
## Requirements

We used Python 3.10.14 in our experiments. To install the necessary dependencies, run:

```sh
pip install -r requirements/requirements_mixat.txt
python -m spacy download en_core_web_sm
```

A Together API token is required to generate PAP samples for MixAT. It must be copied to the corresponding place in `Continuous-AdvTrain/config/harmbench/PAP_config_mixat.yaml`. Additionally, Hugging Face token has to be set in `HF_TOKEN` environmental variable.

## Running Experiments

To execute the experiments on Zephyr, use the following command:

```sh
cd Continuous-AdvTrain
python src/run_experiments.py --config-name=Zephyr_MixAT path=path_zephyr_ul
```

To execute the experiments on  LLama-3-8B, use the following command:

```sh
cd Continuous-AdvTrain
python src/run_experiments.py --config-name=Llama3_MixAT path=path_llama_ul
```


You can modify or add configurations in the `Continuous-AdvTrain/configs` folder to tailor experiments to your needs. By default, trained models are saved to `Continuous-AdvTrain/experiments` We run our training on 2 A100 40G GPUs for Zephyr-7B and 3 A100s 40G GPUs for LLAMA-3-8B or on 1 H200. Qwen-32B trainings were run on 2 H200s. Our code is based on the `Continuous-AdvTrain` [1] repository.

## Robustness Evaluations
To install the necessary dependencies, run:

```sh
pip install -r requirements/requirements_harmbench.txt
python -m spacy download en_core_web_sm
cd harmbench/requirement/FastChat
pip3 install -e ".[model_worker,webui]"
```

To attack generation:

```sh
cd harmbench
./../robustness_eval_harmbench/generate_samples.sh <attack> <model>
```

To evaluate generated attacks:

```sh
cd harmbench
./../robustness_eval_harmbench/evaluate_samples.sh <attack> <model>
```

The attack should be one of the attacks from HarmBench [2] (e.g. GCG). The model should be the name of a model (e.g. zephyr\_7b\_base) which has an entry in `harmbench/configs/model_configs/models.yaml` and in `harmbench/configs/method_configs`, if necessary. Some of the attacks might require Together API or OpenAI API tokens which have to be copied to the corresponding places in `harmbench/configs/method_configs`. Additionally, Hugging Face token and cache path have to be specified in `robustness_eval_harmbench/generate_samples.sh` and `robustness_eval_harmbench/evaluate_samples.sh`.

To inspect the ALO scores of evaluated attacks:

```sh
cd robustness_eval_harmbench
python analyze_results.py
```

This will create a `.md` file with the ALO scores of different attacks for the model specified. Models to inspect have to be specified in `analyze_results.py`.

We used the following attacks for evaluation:

| Attack     |
|----------|
| DirectRequest    |
| PAP      |
| PAIR  |
| TAP  |
| GCG  |
| HumanJailbreaks  |


## XSTest Utility Evaluation
To install the necessary dependencies, run:

```sh
pip install -r requirements/requirements_xstest.txt
python -m spacy download en_core_web_sm
cd harmbench/requirement/FastChat
pip3 install -e ".[model_worker,webui]"
```

To run XSTest [5] utility evaluation (compliance with benign prompts):

```sh
cd harmbench
./../robustness_eval_harmbench/evaluate_xstest_compliance.sh <model>
```

Model is a model's name which has an entry in `harmbench/configs/model_configs/models.yaml`. Additionally, Hugging Face token, Hugging Face cache path and OpenAI API token have to be specified in `robustness_eval_harmbench/eval_commands_compliance.sh`.

To inspect XSTest [5] results:

```sh
cd robustness_eval_harmbench
python analyze_results_xstest.py
```

This will create a `.md` file with the results for the chosen models. Models have to be specified in `robustness_eval_harmbench/analyze_results_xstest.py`.

## HarmLess Utility Evaluation
To install requirements for HarmLess [1] utility evaluation:

```sh
pip install -r requirements/requirements_utility.txt
```

To run HarmLess [1] utility evaluation:

```sh
cd eval_harmless
./run.sh <model_path> <peft_path>(optional) <model_name>
```

Model path and peft path should be a Hugging Face or compatible local path. Peft path is optional, `None` has to be specified explicitly if models without adapters are used. Model name is a custom name for the experiment. Additionally, Hugging Face token and cache path should be specified in `eval_harmless/run.sh`.

## ARC-E, ARC-C and MMLU Utility Evaluation
To install requirements for ARC-E, ARC-C and MMLU [3] utility evaluation:

```sh
pip install -r requirements/requirements_utility.txt
cd lm-evaluation-harness
pip install -e .
```

To run ARC-E, ARC-C and MMLU [3] utility evaluation:

```sh
cd lm-evaluation-harness
./../utility_eval/lm-evaluation-harness/run_utility_test.sh <model_path>
```

Or if a model with adapter:

```sh
cd lm-evaluation-harness
./../utility_eval/lm-evaluation-harness/run_utility_test_peft.sh <model_path> <peft_path>
```

Model path and peft path should be a Hugging Face or compatible local path. Additionally, Hugging Face token and cache path should be specified in `utility_eval/lm-evaluation-harness/run_utility_test.sh` and in `utility_eval/lm-evaluation-harness/run_utility_test_peft.sh`.

## MT-Bench Utility Evaluation
To install requirements for MT-Bench [4] utility evaluation:

```sh
pip install -r "requirements/requirements_utility.txt"
pip install -e "fastchat/[model_worker,llm_judge]"
```

To run MT-Bench [4] utility evaluation:

```sh
cd fastchat/fastchat/llm_judge
./../../../utility_eval/mt_bench/run_utility_evaluate.sh <model_path> <model_name>
```

Or if a model with adapter:

```sh
cd fastchat/fastchat/llm_judge
./../../../utility_eval/mt_bench/run_utility_evaluate_peft.sh <model_path> <model_name> <peft_path>
```

Model path and peft path should be a Hugging Face or compatible local path. Model name is custom name for the experiment. Additionally, Hugging Face token, Hugging Face cache path and OpenAI API token have to be specified in `utility_eval/mt_bench/run_utility_evaluate.sh` and in `utility_eval/mt_bench/run_utility_evaluate_peft.sh`.


## References

Our code is based on a custom version of the following open-source repositories:

[1] [Continuous-AdvTrain](https://github.com/sophie-xhonneux/Continuous-AdvTrain)

[2] [HarmBench](https://github.com/centerforaisafety/HarmBench)

[3] [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

[4] [FastChat](https://github.com/lm-sys/FastChat)

[5] [XSTest](https://github.com/paul-rottger/xstest)

For further details, please refer to the original papers and repositories.

