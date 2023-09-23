import os
import json
from tqdm import tqdm

# Define folder name, data set mapping, and generator mapping
# Define folder name, data set mapping, and generator mapping
folder_name = "embedding_data"
data_set_mapping = {
    1: "arxiv",
    2: "peerread",
    3: "reddit",
    4: "wikihow",
    5: "wikipedia"
}

generator_mapping = {
    1: "bloomz",
    2: "chatGPT",
    3: "cohere",
    4: "davinci",
    5: "dolly"
}

# Function to count vectors in a specific file
def count_vectors_in_file(file_path):
    with open(file_path, 'r') as file:
        count = sum(1 for line in file)
    return count

# Iterate over all combinations of datasets and generators
for data_set_number, data_set_name in data_set_mapping.items():
    for generator_number, generator_name in generator_mapping.items():
        file_name = f"{data_set_name}_{generator_name}_emb.jsonl"
        file_path = os.path.join(folder_name, file_name)
        
        if os.path.exists(file_path):  # Check if the file exists
            vector_count = count_vectors_in_file(file_path)
            print(f"File: {file_name} contains {vector_count} vectors.")
