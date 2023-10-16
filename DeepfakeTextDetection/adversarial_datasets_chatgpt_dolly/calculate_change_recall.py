import os
import json
import csv

def extract_recall_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.loads(file.readline())
        return data.get('recall_machine', None)

def calculate_percent_change(original, new):
    if original == 0:
        return 'NA' 
    return ((new - original) / original) * 100

def main(input_directory, output_directory):
    categories = ['arxiv', 'peerread', 'reddit', 'wikihow', 'wikipedia']
    models = ['chatgpt', 'dolly']
    changes = []

    for category in categories:
        for model in models:
            adv_file = f'EVAL_adversarial_{category}_{model}.jsonl'
            original_file = f'EVAL_{category}_{model}_200.jsonl'

            # Check for file existence
            if not os.path.exists(os.path.join(input_directory, adv_file)) or not os.path.exists(os.path.join(input_directory, original_file)):
                print(f"Files for {category}_{model} not found. Skipping...")
                continue

            original_recall = extract_recall_from_file(os.path.join(input_directory, original_file))
            adversarial_recall = extract_recall_from_file(os.path.join(input_directory, adv_file))

            change = calculate_percent_change(original_recall, adversarial_recall)
            changes.append((adv_file, change))

    output_csv_file = os.path.join(output_directory, 'recall_change_gltrGPT2.csv')
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'Percent Change in Recall'])
        writer.writerows(changes)

    print(f"Output written to {output_csv_file}")

if __name__ == "__main__":
    input_directory = '/workspace/storage/generalized-text-detection/DeepfakeTextDetection/GLTR/eval_results_GPT2'  # Replace with your input directory path
    output_directory = '/workspace/storage/generalized-text-detection/DeepfakeTextDetection/adversarial_datasets_chatgpt_dolly/change_recall_results'  # Replace with your output directory path
    main(input_directory, output_directory)
