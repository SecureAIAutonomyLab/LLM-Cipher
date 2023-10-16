import json
import os
import glob
import numpy as np
from transformers import T5Tokenizer
import argparse

NUM_PROCESSES = 4 
parser = argparse.ArgumentParser(description='Parallel file processing script.')
parser.add_argument('--split', type=int, help='Split number for parallel processing.')
args = parser.parse_args()

split = args.split

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")

def tokenize_text(text):
    if isinstance(text, list):
        text = text[0]
    assert isinstance(text, str), "Text is not a str!"   
    tokenized_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device=0)
    return tokenized_text

source_directory = '../DeepfakeTextDetection/data_In_The_Wild/'
destination_directory = '../data/original_cleaned_adversarial/'

filepaths = glob.glob(f"{source_directory}*.jsonl") 

files_per_split = len(filepaths) // NUM_PROCESSES

start_index = split * files_per_split

end_index = start_index + files_per_split if split != NUM_PROCESSES - 1 else None

filepaths_to_process = filepaths[start_index:end_index]

print(filepaths_to_process)

for filepath in filepaths_to_process:
    print(f'Starting {filepath}')
    
    file_name = os.path.basename(filepath).split('.')[0]

    machine_text = []
    human_text = []

    with open(filepath, 'r') as infile:
        for line_number, line in enumerate(infile):
            if len(line) <= 20:
                continue
            row = json.loads(line)
            print('made it here')
            # Check for the new label and text structure
            if 'label' in row and 'text' in row:
                tokenized_text = tokenize_text(row['text'])
                truncated_text = tokenizer.batch_decode(
                    tokenized_text.input_ids, skip_special_tokens=True)[0]

                if row['label'] == 'machine':
                    machine_text.append(truncated_text)
                elif row['label'] == 'human':
                    human_text.append(truncated_text)
            else:
                tokenized_text = tokenize_text(row['article'])
                truncated_text = tokenizer.batch_decode(
                    tokenized_text.input_ids, skip_special_tokens=True)[0]

                if row['label'] == 'machine':
                    machine_text.append(truncated_text)
                elif row['label'] == 'human':
                    human_text.append(truncated_text)

    with open(destination_directory + f'{file_name}.jsonl', 'a') as json_file:
        for human_row, machine_row in zip(human_text, machine_text):
            json_obj = {
                'human_text' : human_row,
                'machine_text' : machine_row,
            }
            json_str = json.dumps(json_obj)
            json_file.write(json_str + '\n')

print("Complete!")
