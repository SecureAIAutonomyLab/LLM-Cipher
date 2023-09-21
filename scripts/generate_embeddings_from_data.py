import json
import os
import glob
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

encoder = T5EncoderModel.from_pretrained('google/flan-t5-xl').to(device=0)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

def encode_text(text):
    """
    Returns the last hidden state of the given text using the T5 encoder.
    """
    if isinstance(text, list):
        text = text[0]
    assert isinstance(text, str), "Text is not a str!"   
    tokenized_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device=0)
    embeddings = encoder.forward(input_ids=tokenized_text.input_ids, attention_mask=tokenized_text.attention_mask)
    return embeddings['last_hidden_state'][0][-1].tolist()

source_directory = '../data/'
destination_directory = '../embedding_data/' 
allowed_prefixes = ['wikihow_cohere', 'wikipedia', 'peerread', 'reddit', 'arxiv']

os.makedirs(destination_directory, exist_ok=True)

filepaths = sum([glob.glob(f"{source_directory}{prefix}*.jsonl") for prefix in allowed_prefixes], [])

for filepath in filepaths:
    print(f'Starting {filepath}')
    with open(filepath, 'r') as infile:
        filename = os.path.basename(filepath).split('.')[0]
        
        new_filename = os.path.join(destination_directory, filename + '_emb.jsonl')
        
        with open(new_filename, 'w') as outfile:
            for line_number, line in enumerate(infile):
                if len(line) <= 20:
                    continue
                row = json.loads(line)

                human_embedding = None
                machine_embedding = None

                key_pairs = [
                    ('human_text', 'machine_text'),
                    ('text', 'machine_text'),
                    ('abstract', 'machine_abstract'),
                    ('human_reviews', 'bloom_reviews')
                ]

                for human_key, machine_key in key_pairs:
                    if human_key in row and machine_key in row and row[human_key] and row[machine_key]:
                        human_embedding = encode_text(row[human_key])
                        machine_embedding = encode_text(row[machine_key])
                        break

                if not human_embedding or not machine_embedding:
                    continue

                embedding_data = {
                    'human_text_embedding': human_embedding,
                    'machine_text_embedding': machine_embedding
                }

                outfile.write(json.dumps(embedding_data) + '\n')

print("Processing complete!")