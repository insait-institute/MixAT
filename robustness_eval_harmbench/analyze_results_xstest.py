import os
from glob import glob
import json
import numpy as np
import pandas as pd

RUN_ALL = False

models = [
    , #TODO: specify harmbench models here
]

mods = [
    'safe_refusal',
    'safe_partial',
    'safe_compliance',
    'harm_refusal',
    'harm_partial',
    'harm_compliance',
]

def get_results_from_file(file,check_pap=False):
    results = pd.read_csv(file)

    # filter the safe and harmful rows
    safe = results[~results['type'].str.contains('contrast')]
    harm = results[results['type'].str.contains('contrast')]

    results = {}
    if 'gpt4_label' not in safe.columns:
        return {mod:0 for mod in mods}
    # count the number of rows based on column 'gpt4_label'
    results['safe_refusal'] = safe[safe['gpt4_label'].str.contains('full_refusal')].shape[0]
    results['safe_partial'] = safe[safe['gpt4_label'].str.contains('partial_refusal')].shape[0]
    results['safe_compliance'] = safe[safe['gpt4_label'].str.contains('compliance')].shape[0]
    results['harm_refusal'] = harm[harm['gpt4_label'].str.contains('full_refusal')].shape[0]
    results['harm_partial'] = harm[harm['gpt4_label'].str.contains('partial_refusal')].shape[0]
    results['harm_compliance'] = harm[harm['gpt4_label'].str.contains('compliance')].shape[0]

    return results

all_results = {}

res_file = "harmbench_results_xstest.md" if RUN_ALL else "harmbench_results_xstest.md"
final_report = open(res_file, "w")
final_report.write("# HarmBench Results\n\n")
final_report.write(f"| Model | {' | '.join(mods)} |\n")
final_report.write("| --- | --- | --- | --- | --- | --- | --- |\n")

for model in models:
    print(model)
    all_results[model] = {}
    final_report.write(f"| {model} ")
    results_path = f"results/XSTest/completions/{model}.csv"
    if os.path.exists(results_path):
        results = get_results_from_file(results_path)
        for mod in mods:
            final_report.write(f"| {results[mod]} ")
        final_report.write("|\n")
    else:
        print(f"Skipping {model} as results file not found")
        for mod in mods:
            final_report.write(f"| n/a ")
        final_report.write("|\n")
        continue

final_report.close()
print(f"Results saved to {res_file}")

