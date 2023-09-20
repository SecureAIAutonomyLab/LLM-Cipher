import json
import os
import glob
from transformers import T5Config, T5EncoderModel, T5Tokenizer

config = T5Config()
encoder = T5EncoderModel(config)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def encode_text(text):
    """
    Returns the last hidden state of the given text using the T5 encoder.
    """
    tokenized_text = tokenizer(text, return_tensors='pt', padding='max_length', max_length=512)
    embeddings = encoder.forward(input_ids=tokenized_text.input_ids, attention_mask=tokenized_text.attention_mask)
    return embeddings['last_hidden_state'][0][-1].tolist()

source_directory = '../data/'
destination_directory = '../embedding_data/' 
allowed_prefixes = ['wikihow', 'wikipedia', 'peerread', 'reddit', 'arxiv']

os.makedirs(destination_directory, exist_ok=True)

filepaths = sum([glob.glob(f"{source_directory}{prefix}*.jsonl") for prefix in allowed_prefixes], [])

for filepath in filepaths:
    with open(filepath, 'r') as infile:
        filename = os.path.basename(filepath).split('.')[0]
        
        new_filename = os.path.join(destination_directory, filename + '_embedding.jsonl')
        
        with open(new_filename, 'w') as outfile:
            for line in infile:
                row = json.loads(line)
                if 'human_text' in row and 'machine_text' in row:
                    human_embedding = encode_text(row['human_text'])
                    machine_embedding = encode_text(row['machine_text'])

                    embedding_data = {
                        'human_text_embedding': human_embedding,
                        'machine_text_embedding': machine_embedding
                    }

                    outfile.write(json.dumps(embedding_data) + '\n')

print("Processing complete!")
