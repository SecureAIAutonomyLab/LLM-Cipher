import json
import os
import glob
import numpy as np
from transformers import RobertaModel, RobertaTokenizer, BertTokenizer
import argparse

NUM_PROCESSES = 8
NUM_GPUS = 2

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=0, help='Which split of data to process')
args = parser.parse_args()
split = args.split

device = split % NUM_GPUS

encoder = RobertaModel.from_pretrained('roberta-base').to(device)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

source_directory = '../data/original_cleaned_m4/'
destination_directory = '../data/embedding_data_roberta_m4_npy/' 

def encode_text(text):
    """
    Returns the [CLS] embedding of the given text using RoBERTa encoder.
    """
    if isinstance(text, list):
        text = text[0]
    assert isinstance(text, str), "Text is not a str!"   
    tokenized_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device)
    embeddings = encoder(**tokenized_text)
    return embeddings['last_hidden_state'][0][0].detach().cpu().numpy()


def truncate_text(text):
    """
    This function is used for consistency in how we prepare the embeddings.
    """
    tokens = bert_tokenizer.tokenize(text, add_special_tokens=False)
    
    if len(tokens) > 508:  
        tokens = tokens[:508]  
    
    return bert_tokenizer.convert_tokens_to_string(tokens)


os.makedirs(destination_directory, exist_ok=True)

filepaths = glob.glob(f"{source_directory}*.jsonl")

files_per_split = len(filepaths) // NUM_PROCESSES

start_idx = split * files_per_split
end_idx = start_idx + files_per_split if split != (NUM_PROCESSES - 1) else None  # Last split takes any remaining files

filepaths_to_process = filepaths[start_idx:end_idx]

for filepath in filepaths_to_process:
    print(f'Starting {filepath}')
    
    domain_name = os.path.basename(filepath).split('.')[0].split('_')[1]
    model_name =  os.path.basename(filepath).split('.')[0].split('_')[0]

    machine_embeddings = []
    human_embeddings = []

    with open(filepath, 'r') as infile:
        for line_number, line in enumerate(infile):
            if len(line) <= 20:
                continue
            row = json.loads(line)

            human_key = 'human_text'
            machine_key = 'machine_text'

            human_text = row[human_key]
            machine_text = row[machine_key]

            if isinstance(human_text, list):
                for text in human_text:
                    text = truncate_text(text)
                    human_embeddings.append(encode_text(text))
            else:
                human_text = truncate_text(human_text)
                human_embeddings.append(encode_text(human_text))

            if isinstance(machine_text, list):
                for text in machine_text:
                    text = truncate_text(text)
                    machine_embeddings.append(encode_text(text))
            else:
                machine_text = truncate_text(machine_text)
                machine_embeddings.append(encode_text(machine_text))

    np.save(os.path.join(destination_directory, f'{model_name}_{domain_name}_machine_original_cleaned.npy'), np.array(machine_embeddings))
    np.save(os.path.join(destination_directory, f'{model_name}_{domain_name}_human_original_cleaned.npy'), np.array(human_embeddings))


print("Complete!")
