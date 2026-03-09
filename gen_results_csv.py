#!/usr/bin/env python3
import os
import argparse
import glob
import json

from collections import defaultdict, OrderedDict

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint
import pandas as pd

RESULTS_ROOT = "/storage/backup/hei/ttt/flame/results/"
OUT_DIR = "/home/hei/ttt/flame/results"

unsupported_tasks = []
selected_tasks = "hellaswag,lambada_openai,mmlu".split(",")

pretrained_models = [
    "Qwen__Qwen2.5-7B-Instruct",
    "Qwen__Qwen3-8B-Base"
]

def load_data(run_name, data_set):
    results_dir = os.path.join(RESULTS_ROOT, run_name, data_set)
    results_files = glob.glob(os.path.join(results_dir, "results*.json"))
    if len(results_files) != 1:
        raise FileNotFoundError(f"Expected exactly one results*.json file in {results_dir}, found {len(results_files)}: {results_files}")
    results_path = results_files[0]

    out_path = os.path.join(OUT_DIR, f'results_{data_set}.csv') 

    with open(results_path, 'r') as f:
        data = json.load(f)

    results = data.get("results", {})
    flattened_data = OrderedDict()

    if data_set == 'niah':
        # Define the context lengths you want to extract
        context_lengths = ["4096", "8192", "16384", "32768"]
        for key, metrics in results.items():
            # Remove 'niah_' prefix to get 'single_1', 'single_2', etc.
            clean_name = key.replace("niah_", "")
            for length in context_lengths:
                column_name = f"{clean_name}_{length}"
                json_key = f"{length},none"
                if json_key in metrics:
                    flattened_data[column_name] = metrics[json_key] * 100
                else:
                    flattened_data[column_name] = None

    elif data_set == 'lm':
        all_tasks = "gsm8k,winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,race".split(",")
        sorted_tasks = sorted(all_tasks)
        # iterate over sorted task names so columns are added in order
        for task in sorted_tasks:
            if task in results:
                metrics = results[task]
                column_name = task
                json_keys = ["acc,none", "exact_match,strict-match"]
                for json_key in json_keys:
                    if json_key in metrics:
                        flattened_data[column_name] = metrics[json_key] * 100
                        break

    elif data_set == 'lb':
        all_tasks = "2wikimqa,dureader,gov_report,hotpotqa,lcc,lsht,multi_news,multifieldqa_en,multifieldqa_zh,musique,narrativeqa,passage_count,passage_retrieval_en,passage_retrieval_zh,qasper,qmsum,repobench-p,samsum,trec,triviaqa,vcsum".split(",")
        sorted_tasks = sorted(all_tasks)
        # ensure results are inserted in sorted order of task names
        for task in sorted_tasks:
            # results keys have 'longbench_' prefix
            raw_key = f"longbench_{task}"
            if raw_key in results:
                metrics = results[raw_key]
                column_name = task
                json_keys = ["score,none"]
                for json_key in json_keys:
                    if json_key in metrics:
                        flattened_data[column_name] = metrics[json_key] * 100
                        break

    df = pd.DataFrame([flattened_data])
    # df = df.reindex(sorted(df.columns), axis=1)
    # Format all float values to two decimal places using DataFrame.astype(str) and DataFrame.round
    for col in df.select_dtypes(include=["float", "float64"]).columns:
        df[col] = df[col].map(lambda x: f"{x:.2f}")
    df.to_csv(out_path, index=False)

    print(f"CSV created successfully at {out_path}!")
    print(df)


def main():

    parser = argparse.ArgumentParser(description="plot.py")
    parser.add_argument("--run_name", help="run_name", default="",)
    parser.add_argument("--data_set", help="date_str", default="",)
    # parser.add_argument("--csv", help="gen csv", action="store_true")
    # parser.add_argument("--age", type=int, help="Your age", default=30)
    args = parser.parse_args()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_data(run_name=args.run_name, data_set=args.data_set)

if __name__ == "__main__":
    main()