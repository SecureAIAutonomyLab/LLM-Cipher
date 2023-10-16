# Complete script to aggregate results and save to CSV

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
    models = ['bloomz', 'chatgpt', 'cohere', 'davinci', 'dolly']
    
    results = {}
    
    for dataset in datasets:
        for model in models:
            filename = f"EVAL_processed_{dataset}_{model}.jsonl"
            filepath = os.path.join(folder_path, filename)
            
            if os.path.exists(filepath):
                # Read the data from the file
                data = read_jsonl_file(filepath)
                
                # Extracting machine evaluation results along with accuracy, 
                # convert to percentages and round to 2 decimal places
                # Note: Accuracy is now the first metric in the tuple
                machine_results = {
                    "accuracy": round(data["test_acc"] * 100, 2),
                    "precision_machine": round(data["test_precision_machine"] * 100, 2),
                    "recall_machine": round(data["test_recall_machine"] * 100, 2),
                    "fscore_machine": round(data["test_fscore_machine"] * 100, 2)
                }
                
                results[f"{dataset}_{model}"] = machine_results

    # Create a table from the results
    table = pd.DataFrame(index=datasets, columns=models)

    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model}"
            if key in results:
                accuracy = results[key]["accuracy"]
                precision = results[key]["precision_machine"]
                recall = results[key]["recall_machine"]
                f1 = results[key]["fscore_machine"]
                table.loc[dataset, model] = (accuracy, precision, recall, f1)

    # Save the table to a CSV file
    csv_file_path = os.path.join(folder_path, "evaluation_results_RoBERTa-D.csv")
    table.to_csv(csv_file_path)
    print(f"Results saved to {csv_file_path}")


folder_path = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/RoBERTa-Defense/eval_results"
aggregate_results(folder_path)
