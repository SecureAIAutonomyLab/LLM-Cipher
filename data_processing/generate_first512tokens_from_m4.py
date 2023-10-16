import json
import os
import glob
import numpy as np
from transformers import T5Tokenizer
import argparse

NUM_PROCESSES = 28

parser = argparse.ArgumentParser(description='Parallel file processing script.')
parser.add_argument('--split', type=int, help='Split number for parallel processing.')
args = parser.parse_args()

split = args.split

# encoder = T5EncoderModel.from_pretrained('google/flan-t5-xl').to(device=0)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")

def tokenize_text(text):
    if isinstance(text, list):
        text = text[0]
    assert isinstance(text, str), "Text is not a str!"   
    tokenized_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device=0)
    return tokenized_text

source_directory = '../data/original/'
destination_directory = '../data/original_cleaned_inthewild/'

filepaths = glob.glob(f"{source_directory}*.jsonl") 

files_per_split = len(filepaths) // NUM_PROCESSES

start_index = split * files_per_split

end_index = start_index + files_per_split if split != NUM_PROCESSES - 1 else None

filepaths_to_process = filepaths[start_index:end_index]

for filepath in filepaths_to_process:
    print(f'Starting {filepath}')
    
    domain_name = os.path.basename(filepath).split('.')[0].split('_')[0]
    model_name =  os.path.basename(filepath).split('.')[0].split('_')[1]

    machine_text = []
    human_text = []

    with open(filepath, 'r') as infile:
        for line_number, line in enumerate(infile):
            if len(line) <= 20:
                continue
            row = json.loads(line)

            key_pairs = [
                ('human_text', 'machine_text'),
                ('text', 'machine_text'),
                ('abstract', 'machine_abstract'),
                ('human_reviews', 'bloom_reviews')
            ]

            for human_key, machine_key in key_pairs:
                # null check + key_pair check
                if human_key in row and machine_key in row and row[human_key] and row[machine_key]:
                    human_tokenized_text = tokenize_text(row[human_key])
                    machine_tokenized_text = tokenize_text(row[machine_key])

                    human_truncated_text = tokenizer.batch_decode(
                        human_tokenized_text.input_ids, skip_special_tokens=True)
                    machine_truncated_text = tokenizer.batch_decode(
                        machine_tokenized_text.input_ids, skip_special_tokens=True)

                    human_text.append(human_truncated_text[0])
                    machine_text.append(machine_truncated_text[0])
                    break

    with open(destination_directory + f'{model_name}_{domain_name}.jsonl', 'a') as json_file:
        for human_row, machine_row in zip(human_text, machine_text):
            json_obj = {
                'human_text' : human_row,
                'machine_text' : machine_row,
            }
            json_str = json.dumps(json_obj)
            json_file.write(json_str + '\n')


print("Complete!")
