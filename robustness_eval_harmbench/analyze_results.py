import os
from glob import glob
import json
import numpy as np


RUN_ALL = False

models = [
   , #TODO: specify harmbench models here
]



attacks = [
   'DirectRequest',
    'PAP',
    'TAP',
    'PAIR',
    'AutoDAN',
    'GCG',
   'HumanJailbreaks',
]

def get_results_from_file(file,check_pap=False):
    with open(file, 'r') as f:
        results = json.load(f)
    filtered_results = {}
    filtered_results['pap_check'] = True
    for k, v in results.items():
        filtered_results[k] = 0
        if check_pap and len(v) != 10:
            filtered_results['pap_check'] = False
        for item in v:
            if item['label'] == 1:
                filtered_results[k] = 1
                break
    return filtered_results

if RUN_ALL:
    base_path = "samples/all_samples.json" 
else:
    base_path = "samples/samples.json"
base_results = get_results_from_file(base_path)
base_results.pop('pap_check')
all_test_cases = sorted(list(base_results.keys()))
print(len(all_test_cases))
all_results = {}

res_file = "harmbench_results_all.md" if RUN_ALL else "harmbench_results.md"
final_report = open(res_file, "w")
final_report.write("# HarmBench Results\n\n")
final_report.write(f"| Model | {' | '.join(attacks)} | ALO |\n")
final_report.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")

for model in models:
    print(model)
    all_results[model] = {}
    all_results[model]['ALO'] = {k:0 for k in all_test_cases}
    final_report.write(f"| {model} ")
    for attack in attacks:
        all_results[model][attack] = {k:0 for k in all_test_cases}
        if RUN_ALL:
            path = f"results/{attack}/{model}/results_all/{model}.json*"
        else:
            path = f"results/{attack}/{model}/results/{model}.json*" 
        files = glob(path)
        results_file = "missing"
        last_checkpoint = 0
        for file in files:
            if file.endswith(".json"):
                results_file = file
                break
            checkpoint = int(file.split("_")[-1])
            if checkpoint > last_checkpoint:
                last_checkpoint = checkpoint
                results_file = file
        if results_file == "missing":
            if attack in ['DirectRequest', 'PAP']:
                new_path = f"results/{attack}/zephyr_7b_all_models/results/{model}.json"
                if os.path.exists(new_path):
                    results_file = new_path

        if results_file == "missing":
            print(f' {attack:<13}',results_file)
            final_report.write("| / ")
            continue
        results = get_results_from_file(results_file, check_pap=(attack == 'PAP'))
        pap_str = ""
        if not results['pap_check']:
            pap_str = " (!)"
        results.pop('pap_check')
        asr = 0
        complete = True
        good = 0
        for k in all_test_cases:
            if k not in results:
                complete = False
            else:
                good += 1
        if not complete:
            print(f' {attack:<13} (incomplete: {len(results)}) {results_file}')
            pap_str = f" ({good})"
        for k, v in results.items():
            if k not in all_test_cases:
                continue
            all_results[model][attack][k] = v
            asr += v
            if v == 1:
                all_results[model]['ALO'][k] = 1
        print(f' {attack:<13} {results_file}')
        print(f' {attack:<13} {asr:2}/{len(all_test_cases)} = {100*asr/len(all_test_cases):.1f}')
        final_report.write(f"| {100*asr/len(all_test_cases):.1f}{pap_str} ")
    asr = sum(all_results[model]['ALO'].values())
    print(f' {"ALO":<13} {asr:2}/{len(all_test_cases)} = {100*asr/len(all_test_cases):.1f}')
    final_report.write(f"| {100*asr/len(all_test_cases):.1f} |\n")
    print()

final_report.close()
print(f"Results saved to {res_file}")