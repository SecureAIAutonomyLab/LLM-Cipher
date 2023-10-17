# Complete script to aggregate results and save to CSV, can adapt for different dataset evaluations you want to compile

import os
import json
import pandas as pd

def read_jsonl_file(filename):
    """Read a .jsonl file and return the parsed JSON data."""
    with open(filename, 'r') as f:
        line = f.readline()
        data = json.loads(line)
    return data

def aggregate_results(folder_path):
    datasets = ['arxiv', 'peerread', 'reddit', 'wikihow', 'wikipedia']
    models = ['chatgpt', 'dolly']
    
    results = {}
    
    for dataset in datasets:
        for model in models:
            filename = f"EVAL_adversarial_{dataset}_{model}.jsonl"
            filepath = os.path.join(folder_path, filename)
            
            if os.path.exists(filepath):
                # Read the data from the file
                data = read_jsonl_file(filepath)
                
                # Extracting machine evaluation results for f1 score
                machine_results = {
                    "fscore_machine": round(data["eval_fscore_machine"] * 100, 2)
                }
                
                results[f"{dataset}_{model}"] = machine_results

    # Create a table from the results
    table = pd.DataFrame(index=datasets, columns=models)

    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model}"
            if key in results:
                f1 = results[key]["fscore_machine"]
                table.loc[dataset, model] = (f1)

    # Save the table to a CSV file
    csv_file_path = os.path.join(folder_path, "evaluation_results.csv")
    table.to_csv(csv_file_path)
    print(f"Results saved to {csv_file_path}")


folder_path = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/BERT-Defense/eval_results"
aggregate_results(folder_path)
