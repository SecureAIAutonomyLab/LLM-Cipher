import os
import json
import csv

def calculate_evasion_rate(recall, total_machine_generated_samples=200):
    TP = recall * total_machine_generated_samples
    FN = total_machine_generated_samples - TP
    evasion_rate = FN / total_machine_generated_samples
    return evasion_rate

def extract_evasion_rates(input_dir, output_file):
    evasion_rates = []

    # Iterate over files in the input directory
    for file in os.listdir(input_dir):
        if file.startswith("EVAL_adversarial_") and file.endswith(".jsonl"):
            with open(os.path.join(input_dir, file), 'r') as f:
                data = json.load(f)
                recall = data.get("test_recall_machine", 0)
                evasion_rate = calculate_evasion_rate(recall)
                evasion_rates.append((file, evasion_rate))

    # Save evasion rates to CSV
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["File Name", "Evasion Rate"])
        csv_writer.writerows(evasion_rates)

# Set your directories here
parent_directory = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/RoBERTa-Defense/eval_results"
output_directory = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/adversarial_datasets_chatgpt_dolly/evasion_rate_results"
output_file = os.path.join(output_directory, "evasion_rates_robertaD.csv")

extract_evasion_rates(parent_directory, output_file)
