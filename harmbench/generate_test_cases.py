import transformers

from baselines import get_method_class, init_method
import yaml
import argparse
import csv
import torch
import os
from os.path import join
from eval_utils import get_experiment_config
import pandas as pd
import gc
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()
import copy
import ray
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--method_name", type=str, default='GCG',
                        help="The name of the red teaming method (options are in baselines/__init__.py)")
    parser.add_argument("--experiment_name", type=str,
                        help="The name of the experiment (options are in configs/methods_config/{{method_name}}_config.yaml)")
    parser.add_argument("--method_config_file", type=str, default=None,
                        help="The path to the config file with method hyperparameters. This will default to configs/method_configs/{{method_name}}_config.yaml")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--save_dir", type=str, default='./test_cases',
                        help="The directory used for saving test cases")
    parser.add_argument("--behavior_start_idx", type=int, default=None,
                        help="Start index for behaviors_path data (inclusive)")
    parser.add_argument("--behavior_end_idx", type=int, default=None,
                        help="End index for behaviors_path data (exclusive)")
    parser.add_argument("--lora_scaling_experiments", action="store_true",
                        help="Used when experimenting with scaling the LORA weights")
    parser.add_argument("--behavior_ids_subset", type=str, default='',
                        help="An optional comma-separated list of behavior IDs, or a path to a newline-separated list of behavior IDs. If provided, this will override behavior_start_idx and behavior_end_idx for selecting a subset of behaviors.")
    parser.add_argument("--run_id", type=int, default=None,
                        help="An optional run id for output files, will be appended to output filenames. Must be an integer.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing saved test cases")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print out intermediate results")
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
        default=0,
        type=int,
        help="Final checkpoint number",
    )

    parser.add_argument(
        "--list_num",
        default=None,
        type=str,
        help="Numbers separated with comas. If provided this fill be the interval",
    )
    
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    method_name = args.method_name
    experiment_name = args.experiment_name

    save_dir = os.path.abspath(args.save_dir)

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    print("Experiment Model Config:", model_configs[experiment_name])
    # Load method config file
    config_file = f"configs/method_configs/{method_name}_config.yaml" if not args.method_config_file else args.method_config_file
    with open(config_file) as file:
        method_configs = yaml.full_load(file)
    
    # Load default method parameters
    if 'default_method_hyperparameters' not in method_configs:
        raise ValueError(f"Default method hyperparameters not found in {config_file}")
    method_config = method_configs.get('default_method_hyperparameters') or {}

    # Update method parameters with experiment parameters
    experiment_config = get_experiment_config(experiment_name, model_configs, method_configs)
    method_config_mult = copy.deepcopy(method_config)
    method_config.update(experiment_config)


    if args.run_id:
        assert args.run_id.isdigit(), f"epxected run_id to be an integer; got {args.run_id}"
    
    # ========== generate test cases ========== #
    # load behaviors csv
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)

    if args.behavior_ids_subset:
        print(f"Selecting subset from provided behavior IDs: {args.behavior_ids_subset}")
        if args.behavior_ids_subset.startswith('no_copy'):
            behaviors = [b for b in behaviors if b['FunctionalCategory'] != 'copyright']
        else:
            if os.path.exists(args.behavior_ids_subset):
                with open(args.behavior_ids_subset, 'r') as f:
                    behavior_ids_subset = f.read().splitlines()
            else:
                behavior_ids_subset = args.behavior_ids_subset.split(',')
            behavior_ids_subset = list(map(int, behavior_ids_subset))
            behaviors = [behaviors[i] for i in behavior_ids_subset]
    else:
        behavior_start_idx, behavior_end_idx = args.behavior_start_idx, args.behavior_end_idx
        if args.behavior_start_idx is not None and args.behavior_end_idx is not None:
            print(f"Selecting subset from [{behavior_start_idx}, {behavior_end_idx}]")
            behaviors = behaviors[args.behavior_start_idx:args.behavior_end_idx]


    world_size = torch.cuda.device_count()

    print(f"============= {method_name}, {experiment_name} Config =============")
    print('\n'.join([f"{k}: {v}" for k, v in method_config.items()]))
    print("============= GPU Allocation Configs =============")
    print(f"Total GPUs: {world_size}")
    print("=============")

    # ==== Filtering existed runs =====
    method_class = get_method_class(method_name)

    if not args.overwrite:
        filtered_behaviors = []
        i_temp_i = []
        for i,behavior in enumerate(behaviors):
            # flatten csv row
           
            behavior_id = behavior['BehaviorID']
            test_case_file = method_class.get_output_file_path(save_dir, behavior_id, file_type='test_cases', run_id=args.run_id)

            if not os.path.exists(test_case_file) or method_name=="DirectRequest":
                filtered_behaviors.append(behavior)
                i_temp_i.append(i)
            else:
                print(f"===>Skipping {behavior_id}")
        behaviors = filtered_behaviors

    # ====== Run attack ======
    print("#Behaviors to run:" , len(behaviors))
    assert (not args.lora_scaling_experiments) or  (not args.multiple_checkpoints), "ERROR"
    if args.lora_scaling_experiments:
        print("RANGE")
        if args.list_num:
            interval = args.list_num.split(",")
            interval = [int(num)/100 for num in interval]
        else:
            interval = [-0.5,0.0,0.5,1.0,1.5]  
        print("interval")
        print(interval)
        

        for i,lam in enumerate(interval):
            # ==== Init Method ===
            method_config_new = copy.deepcopy(method_config)
            method_class_new = copy.deepcopy(method_class)
            method_config_new["lam"] = lam

            method = init_method(method_class_new, method_config_new)
            save_dir_new = save_dir + f"_lambda_{lam}"
            filtered_behaviors = []
            for behavior in behaviors:
                # flatten csv row
                behavior_id = behavior['BehaviorID']
                test_case_file = method_class.get_output_file_path(save_dir_new, behavior_id, file_type='test_cases', run_id=args.run_id)
                if not os.path.exists(test_case_file) or method_name=="DirectRequest":
                    filtered_behaviors.append(behavior)
                else:
                    print(f"===>Skipping {behavior_id}")
            behaviors_new = filtered_behaviors

            test_cases, logs = method.generate_test_cases(behaviors=behaviors_new, verbose=args.verbose,save_dir=save_dir_new,method_config=method_config_new,run_id=args.run_id)
            # ====== Save test cases and logs =====
            print("Saving outputs to: ", save_dir)
            if method_name == "DirectRequest":
                method.save_test_cases(save_dir_new, test_cases, logs, method_config=method_config_new, run_id=args.run_id)
            print("Finished")
            del method
    elif args.multiple_checkpoints:

        interval = list(range(args.begin,args.end,args.step))
        if args.final != 0:
            interval += [args.final]
        if args.list_num:
            interval = args.list_num.split(",")
            interval = [int(num) for num in interval]
        print("interval")
        print(interval)
        for i,lam in enumerate(interval):
            # ==== Init Method ====
            method_config_new = copy.deepcopy(method_config_mult)
            method_config_new.update(experiment_config)
            method_config_new["target_model"]["peft_path"] = method_config_new["target_model"]["peft_path"] + f"/trainer_{lam}.0_epoch"
            print(f"============= {method_name}, {experiment_name} Config =============")
            print('\n'.join([f"{k}: {v}" for k, v in method_config_new.items()]))
            print("============= GPU Allocation Configs =============")
            print(f"Total GPUs: {world_size}")
            print("=============")
            method = init_method(method_class, method_config_new)
            save_dir_new = save_dir + f"_checkpoint_{lam}"
            test_cases, logs = method.generate_test_cases(behaviors=behaviors, verbose=args.verbose,save_dir=save_dir_new,method_config=method_config_new,run_id=args.run_id)
            if method_name == "DirectRequest":
                method.save_test_cases(save_dir_new, test_cases, logs, method_config=method_config_new, run_id=args.run_id)

    else:
        # ==== Init Method ====
        old_method_config = copy.deepcopy(method_config)
        method = init_method(method_class, method_config)    
        test_cases, logs = method.generate_test_cases(behaviors=behaviors, verbose=args.verbose,save_dir=save_dir,method_config=method_config,run_id=args.run_id)
        
        # ====== Save test cases and logs ======
    
        if method_name == "DirectRequest" or method_name == "HumanJailbreaks":
            method.save_test_cases(save_dir, test_cases, logs, method_config=method_config, run_id=args.run_id)

if __name__ == "__main__":
    main()
