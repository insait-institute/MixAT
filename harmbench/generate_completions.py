import transformers
import json
import argparse
import os, gc
import csv
from tqdm import tqdm 
import torch
from baselines import get_template, load_model_and_tokenizer, load_vllm_model
from api_models import api_models_map
from functools import partial
import multimodalmodels
from PIL import Image
from torchvision import transforms
from vllm import SamplingParams
from accelerate.utils import find_executable_batch_size

import copy
import gc
import torch
import yaml
import pandas as pd


# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--model_name", type=str,
                        help="The name of the model in the models config file")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--test_cases_path", type=str,
                        help="The path to the test cases file to generate completions for")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving completions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for completions")
    parser.add_argument("--generate_with_vllm", action="store_true",
                        help="Whether to generate completions with vLLM (if applicable)")
    parser.add_argument("--incremental_update", action="store_true",
                        help="Whether to incrementally update completions or generate a new completions file from scratch")
    parser.add_argument("--lora_scaling_experiments", action="store_true",
                        help="Used when experimenting with scaling the LORA weights")
    parser.add_argument("--temperature_repeats", type=int, default=0,
                        help="Use for generating completions with different temperatures. Set to number of repeats for each test case.")
    parser.add_argument(
        "--multiple_checkpoints",
        action="store_true",
        help="Whether doing multiple checkpoint experiments or not",
    )
    parser.add_argument(
        "--begin",
        type=int,
        help="Starting of checkpoint",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="Ending of checkpoint",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Checkpoint exmination step",
    )
    parser.add_argument(
        "--final",
        type=int,
        help="Final checkpoint number",
    )
    parser.add_argument(
        "--list_num",
        default=None,
        type=str,
        help="Numbers separated with comas. If provided this fill be the interval",
    )
    parser.add_argument("--filter", type=str, choices=['all', 'harmless', 'harmful'], default='all',
                        help="Filter prompts based on their type")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)
    
    num_gpus = model_configs[args.model_name].get('num_gpus', 0)
    # check num gpus available to see if greater than num_gpus in config
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs.")
        num_gpus = num_gpus_available
    model_config = model_configs[args.model_name]['model']
    model_config['num_gpus'] = num_gpus

    print("model_config", model_config)
    
    # ========== generate test cases ========== #
    # load behaviors csv
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        behaviors = list(reader)[1:]  # ignore the header
    
    assert (not args.lora_scaling_experiments) or  (not args.multiple_checkpoints), "Can not  be these two at the same time"
    if args.lora_scaling_experiments:
        print("RANGE")
        if args.list_num:
            if args.list_num.startswith("m"):
                interval = args.list_num[1:].split(",")
                interval = [int(num)/100*(-1) for num in interval]
            else:
                interval = args.list_num.split(",")
                interval = [int(num)/100 for num in interval]
        else:
            interval = [-0.5,0.0,0.5,1.0,1.5]
        print("interval")
        print(interval)
        for i,lam in enumerate(interval):
            gc.collect()
            torch.cuda.empty_cache()
            if args.test_cases_path.endswith(".csv"):
                test_cases_path_new = args.test_cases_path
                prompts_df = pd.read_csv(args.test_cases_path)

                # Filter prompts based on the --filter argument
                if args.filter == 'harmless':
                    prompts_df = prompts_df[~prompts_df['type'].str.startswith('contrast')]
                elif args.filter == 'harmful':
                    prompts_df = prompts_df[prompts_df['type'].str.startswith('contrast')]

                prompts = prompts_df['prompt'].tolist()
                test_cases = [{'behavior_id': idx, 'test_case': prompt} for idx, prompt in enumerate(prompts)]

            else:
                splitted = args.test_cases_path.split('/')
                splitted[-2] += f"_lambda_{lam}"
                test_cases_path_new = '/'.join(splitted)
                with open(test_cases_path_new, 'r') as f:
                    test_cases_data = json.load(f)

                returned_data = {}

                if args.incremental_update and os.path.exists(args.save_path):
                    with open(args.save_path, 'r') as file:
                        returned_data = json.load(file)
                    
                    # Remove test cases that have already been generated
                    new_test_cases_data = {}
                    new_returned_data = {}
                    for behavior_id, _test_cases in test_cases_data.items():
                        # Check if behavior_id is in returned_data and completions have been generated for all test cases
                        if behavior_id in returned_data and len(returned_data[behavior_id]) == len(_test_cases):
                            continue
                        new_test_cases_data[behavior_id] = _test_cases
                        new_returned_data[behavior_id] = []  # If not all completions are present, we will regenerate all completions
                    test_cases_data = new_test_cases_data

                # ==== Flatten dict to array ======
                test_cases = []
                for behavior_id, _test_cases in test_cases_data.items():
                    test_cases.extend([dict(behavior_id=behavior_id, test_case=t) for t in _test_cases])

            if len(test_cases) == 0:
                print('No test cases to generate completions for')
                return
            else:
                print(f'Generating completions for {len(test_cases)} test cases')
            # ==== Generation function ====
            model_config["lam"] = lam
            try:
                generation_function = load_generation_function(model_config,
                                                    args.max_new_tokens,
                                                    test_cases_path=test_cases_path_new,
                                                    generate_with_vllm=args.generate_with_vllm)
            except Exception as e:
                print(f"Error in loading generation function, retrying with final model instead")
                model_config["peft_path"] = model_config["peft_path"] + f"/final_model"
                generation_function = load_generation_function(model_config,
                                                    args.max_new_tokens,
                                                    test_cases_path=test_cases_path_new,
                                                    generate_with_vllm=args.generate_with_vllm)
            # ==== Generate ====
            print('Generating completions...')
            generations = generation_function(test_cases=[t['test_case'] for t in test_cases])
            print('Done')

            if args.test_cases_path.endswith(".csv"):
                prompts_df['completion'] = generations

                save_path_new = args.save_path + f"_lambda_{lam}"
                
                # Save results to CSV
                print(f'Saving completions to {save_path_new}...')
                os.makedirs(os.path.dirname(save_path_new), exist_ok=True) if os.path.dirname(save_path_new) else None
                prompts_df.to_csv(save_path_new, index=False)
                print('Saved')
            else:
                # ==== Unflatten into dict of {behavior_id: [{test_case: t, generation: g}, ..], ...}
                for t, g in zip(test_cases, generations):
                    returned_data.setdefault(t['behavior_id'], [])
                    returned_data[t['behavior_id']].append(dict(test_case=t['test_case'], generation=g))

                # Create directories for save_path
                save_path_new = args.save_path + f"_lambda_{lam}"
                os.makedirs(os.path.dirname(save_path_new), exist_ok=True) if os.path.dirname(save_path_new) else None
                with open(save_path_new, 'w') as file:
                    json.dump(returned_data, file, indent=4)
                print('Saved')
            gc.collect()
            torch.cuda.empty_cache()

    elif args.multiple_checkpoints:
        interval = list(range(args.begin,args.end,args.step))
        interval += [args.final]
        if args.list_num:
            interval = args.list_num.split(",")
            interval = [int(num) for num in interval]
        print("interval")
        print(interval)
        for i,lam in enumerate(interval):
            splitted = args.test_cases_path.split('/')
            splitted[-2] += f"_checkpoint_{lam}"
            test_cases_path_new = '/'.join(splitted)

            with open(test_cases_path_new, 'r') as f:
                test_cases_data = json.load(f)

            returned_data = {}

            if args.incremental_update and os.path.exists(args.save_path):
                with open(args.save_path, 'r') as file:
                    returned_data = json.load(file)
                
                # Remove test cases that have already been generated
                new_test_cases_data = {}
                new_returned_data = {}
                for behavior_id, _test_cases in test_cases_data.items():
                    # Check if behavior_id is in returned_data and completions have been generated for all test cases
                    if behavior_id in returned_data and len(returned_data[behavior_id]) == len(_test_cases):
                        continue

                    new_test_cases_data[behavior_id] = _test_cases
                    new_returned_data[behavior_id] = []  # If not all completions are present, we will regenerate all completions
                test_cases_data = new_test_cases_data

            # ==== Flatten dict to array ======
            test_cases = []
            for behavior_id, _test_cases in test_cases_data.items():
                test_cases.extend([dict(behavior_id=behavior_id, test_case=t) for t in _test_cases])

            if len(test_cases) == 0:
                print('No test cases to generate completions for')
                return
            else:
                print(f'Generating completions for {len(test_cases)} test cases')
            # ==== Generation function ====
            model_congig_new = copy.deepcopy(model_config)
            model_congig_new["peft_path"] = model_congig_new["peft_path"] + f"/trainer_{lam}.0_epoch"
            generation_function = load_generation_function(model_congig_new,
                                                args.max_new_tokens,
                                                test_cases_path=test_cases_path_new,
                                                generate_with_vllm=args.generate_with_vllm)
            # ==== Generate ====
            print('Generating completions...')
            generations = generation_function(test_cases=[t['test_case'] for t in test_cases])
            print('Done')

            # ==== Unflatten into dict of {behavior_id: [{test_case: t, generation: g}, ..], ...}
            for t, g in zip(test_cases, generations):
                returned_data.setdefault(t['behavior_id'], [])
                returned_data[t['behavior_id']].append(dict(test_case=t['test_case'], generation=g))

            # Create directories for save_path
            save_path_new = args.save_path + f"_checkpoint_{lam}"
            print(f'Saving completions to {save_path_new}...')
            os.makedirs(os.path.dirname(save_path_new), exist_ok=True) if os.path.dirname(save_path_new) else None
            with open(save_path_new, 'w') as file:
                json.dump(returned_data, file, indent=4)
            print('Saved')
            gc.collect()
            torch.cuda.empty_cache()
    
    else:
        if args.test_cases_path.endswith(".csv"):
            prompts_df = pd.read_csv(args.test_cases_path)

            # Filter prompts based on the --filter argument
            if args.filter == 'harmless':
                prompts_df = prompts_df[~prompts_df['type'].str.startswith('contrast')]
            elif args.filter == 'harmful':
                prompts_df = prompts_df[prompts_df['type'].str.startswith('contrast')]

            prompts = prompts_df['prompt'].tolist()
            test_cases = [{'behavior_id': idx, 'test_case': prompt} for idx, prompt in enumerate(prompts)]

        else:
            with open(args.test_cases_path, 'r') as f:
                test_cases_data = json.load(f)

            returned_data = {}

            if args.incremental_update and os.path.exists(args.save_path):
                with open(args.save_path, 'r') as file:
                    returned_data = json.load(file)
                
                # Remove test cases that have already been generated
                new_test_cases_data = {}
                new_returned_data = {}
                for behavior_id, _test_cases in test_cases_data.items():
                    # Check if behavior_id is in returned_data and completions have been generated for all test cases
                    if behavior_id in returned_data and len(returned_data[behavior_id]) == len(_test_cases):
                        continue
                    new_test_cases_data[behavior_id] = _test_cases
                    new_returned_data[behavior_id] = []  # If not all completions are present, we will regenerate all completions
                test_cases_data = new_test_cases_data

            # ==== Flatten dict to array ======
            test_cases = []
            for behavior_id, _test_cases in test_cases_data.items():
                test_cases.extend([dict(behavior_id=behavior_id, test_case=t) for t in _test_cases]) 

        if len(test_cases) == 0:
            print('No test cases to generate completions for')
            return
        else:
            print(f'Generating completions for {len(test_cases)} test cases')
            # ==== Generation function ====
        do_sample = args.temperature_repeats > 0
        try:
            generation_function = load_generation_function(model_config,
                                                        args.max_new_tokens,
                                                        test_cases_path=args.test_cases_path,
                                                        generate_with_vllm=args.generate_with_vllm, do_sample=do_sample)
        except Exception as e:
            print(f"Error in loading generation function, retrying with final model instead")
            model_config["peft_path"] = model_config["peft_path"] + f"/final_model"
            generation_function = load_generation_function(model_config,
                                                        args.max_new_tokens,
                                                        test_cases_path=args.test_cases_path,
                                                        generate_with_vllm=args.generate_with_vllm, do_sample=do_sample)
        if args.temperature_repeats:
            if args.list_num:
                if args.list_num.startswith("m"):
                    interval = args.list_num[1:].split(",")
                    interval = [int(num)/10*(-1) for num in interval]
                else:
                    interval = args.list_num.split(",")
                    interval = [int(num)/10 for num in interval]
            else:
                interval = [i/10 for i in range(1,16)]
        else:
            interval = [0]

        extra_args = {}

        for temp in interval:
            # ==== Generate ====
            print(f'Generating completions temperature = {temp} ...')
            new_test_cases = test_cases
            new_save_path = args.save_path
            if temp != 0:
                extra_args["temperature"] = temp
                extra_args["top_p"] = 0.9
                new_save_path = args.save_path + f"_temp_{temp}"
                new_test_cases = []
                for t in test_cases:
                    for i in range(args.temperature_repeats):
                        new_test_cases.append(copy.deepcopy(t))
            generations = generation_function(test_cases=[t['test_case'] for t in new_test_cases],**extra_args) 
            print('Done')

            if args.test_cases_path.endswith(".csv"):
                prompts_df['completion'] = generations

                # Save results to CSV
                print(f'Saving completions to {new_save_path}...')
                os.makedirs(os.path.dirname(new_save_path), exist_ok=True) if os.path.dirname(new_save_path) else None
                prompts_df.to_csv(new_save_path, index=False)
                print('Saved')
            else:
                # ==== Unflatten into dict of {behavior_id: [{test_case: t, generation: g}, ..], ...}
                for t, g in zip(new_test_cases, generations):
                    returned_data.setdefault(t['behavior_id'], [])
                    returned_data[t['behavior_id']].append(dict(test_case=t['test_case'], generation=g))

                # Create directories for save_path
                print(f'Saving completions to {new_save_path}...')
                os.makedirs(os.path.dirname(new_save_path), exist_ok=True) if os.path.dirname(new_save_path) else None
                with open(new_save_path, 'w') as file:
                    json.dump(returned_data, file, indent=4)
                print('Saved')


def _vllm_generate(model, test_cases, template, **generation_kwargs):
    inputs = [template['prompt'].format(instruction=s) for s in test_cases]
    outputs = model.generate(inputs, **generation_kwargs)
    generations = [o.outputs[0].text.strip() for o in outputs]
    return generations

def _hf_generate_with_batching(model, tokenizer, test_cases, template, **generation_kwargs):
    starting_bs = min(120, len(test_cases))
    @find_executable_batch_size(starting_batch_size=starting_bs)
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, template, generation_kwargs
        generations = []
        for i in tqdm(range(0, len(test_cases), batch_size)):
            batched_test_cases = test_cases[i:i+batch_size]

            inputs = [template['prompt'].format(instruction=s) for s in batched_test_cases]

            inputs = tokenizer(inputs, return_tensors='pt', padding=True)
 
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'], **generation_kwargs).cpu()

            generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            batch_generations = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in generated_tokens]
            generations.extend(batch_generations)
        return generations
    return inner_generation_loop()

def _api_model_generate(model, test_cases, **generation_kwargs):
    # MultiModal test cases are in saved in pair of [[img, text], ...]
    if isinstance(test_cases[0], (tuple, list)):
        images, prompts = zip(*test_cases)
        return model.generate(prompts=list(prompts), images=list(images), **generation_kwargs)   
    return model.generate(prompts=test_cases, **generation_kwargs)

def load_generation_function(model_config, max_new_tokens, test_cases_path, generate_with_vllm=False, do_sample=False):
    model_name_or_path = model_config['model_name_or_path']

    generate_with_vllm = False

    if model_name_or_path in dir(multimodalmodels):
        model = getattr(multimodalmodels, model_name_or_path)()
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        image_dir = os.path.join(os.path.dirname(test_cases_path), 'images')
        return partial(model.generate, image_dir=image_dir, **generation_kwargs)
    if (model := api_models_map(**model_config)):
        generation_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True)
        return partial(_api_model_generate, model=model, **generation_kwargs)
    elif generate_with_vllm:
        print('Using vLLM generation')
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        model = load_vllm_model(**model_config)
        generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        return partial(_vllm_generate, model=model, template=TEMPLATE, **generation_kwargs)
    else:
        print('Using HF generation')
        model, tokenizer = load_model_and_tokenizer(**model_config)
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        print("TEMPLATE")
        print(TEMPLATE)
        return partial(_hf_generate_with_batching, model=model, tokenizer=tokenizer, template=TEMPLATE, **generation_kwargs)

if __name__ == "__main__":
    main()
