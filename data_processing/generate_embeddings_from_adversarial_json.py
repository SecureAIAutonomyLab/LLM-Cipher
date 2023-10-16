import json
import os
import glob
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer, BertTokenizer
import argparse
import torch.nn.functional as F
import torch
import random
import string

source_directory = '/workspace/storage/generalized-text-detection/DeepfakeTextDetection/DFTFooler/adversarial_datasets_ITW/'
destination_directory = '../embedding_data_T5_adversarial_json/'

NUM_PROCESSES = 1
NUM_GPUS = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=0,
                    help='Which split of data to process')
args = parser.parse_args()
split = args.split

device = split % NUM_GPUS

encoder = T5EncoderModel.from_pretrained('google/flan-t5-xl').to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def encode_text(text):
    """
    Returns the last hidden state of the given text using T5 encoder.
    """
    if isinstance(text, list):
        text = text[0]
    assert isinstance(text, str), "Text is not a str!"
    tokenized_text = tokenizer(
        text, return_tensors='pt', max_length=512, truncation=True).to(device)
    embeddings = encoder.forward(
        input_ids=tokenized_text.input_ids, attention_mask=tokenized_text.attention_mask)
    return embeddings['last_hidden_state'][0][-1].detach().cpu().numpy()


def truncate_text(text):
    """
    This function is used for consistency in how we prepare the embeddings.
    """
    tokens = bert_tokenizer.tokenize(text, add_special_tokens=False)

    if len(tokens) > 508:
        tokens = tokens[:508]

    return bert_tokenizer.convert_tokens_to_string(tokens)


def generate_unique_id(length=20):
    return ''.join(random.choice(string.ascii_letters + string.digits) for i in range(length))

os.makedirs(destination_directory, exist_ok=True)

filepaths = glob.glob(f"{source_directory}*.jsonl")

files_per_split = len(filepaths) // NUM_PROCESSES

start_idx = split * files_per_split
end_idx = start_idx + files_per_split if split != (NUM_PROCESSES - 1) else None

filepaths_to_process = filepaths[start_idx:end_idx]

for filepath in filepaths_to_process:
    print(f'Starting {filepath}')

    file_name = os.path.basename(filepath).split('.')[0]

    results = []

    with open(filepath, 'r') as infile:
        for line_number, line in enumerate(infile):
            if len(line) <= 20:
                continue
            row = json.loads(line)

            machine_text = row['text']

            machine_text_truncated = truncate_text(machine_text)

            machine_embedding = encode_text(machine_text_truncated)

            results.append({
                'machine_text': machine_text,
                'machine_embedding': ', '.join(map(str, machine_embedding.tolist())),
                'source_id': generate_unique_id()
            })

    with open(os.path.join(
            destination_directory,
            f'{file_name}.json'), 'w') as outfile:
        json.dump(results, outfile, indent=4)

print("Complete!")
