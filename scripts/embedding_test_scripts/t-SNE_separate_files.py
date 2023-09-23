import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm 

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

# Ensure the 'figures' directory exists
if not os.path.exists("figures"):
    os.makedirs("figures")

# Function to process and visualize t-SNE for a specific file
def process_and_visualize(file_path, data_set_name, generator_name):
    embeddings = []
    labels = []

    # Load embeddings from file
    with open(file_path, "r") as file:
        for line in tqdm(file, desc=f"Processing {os.path.basename(file_path)}", leave=False):
            data = json.loads(line)

            # Append human and machine embeddings
            embeddings.append(data['human_text_embedding'])
            labels.append(f"Human-{data_set_name}")
            
            embeddings.append(data['machine_text_embedding'])
            labels.append(f"Machine-{data_set_name}-{generator_name}")

    # Convert list to numpy array
    X = np.array(embeddings)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Human class
    plt.scatter(X_tsne[np.array(labels) == f"Human-{data_set_name}", 0], 
                X_tsne[np.array(labels) == f"Human-{data_set_name}", 1], 
                label=f"Human-{data_set_name}")

    # Machine class
    plt.scatter(X_tsne[np.array(labels) == f"Machine-{data_set_name}-{generator_name}", 0], 
                X_tsne[np.array(labels) == f"Machine-{data_set_name}-{generator_name}", 1], 
                label=f"Machine-{data_set_name}-{generator_name}")

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f't-SNE visualization for {data_set_name} with {generator_name}')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join("figures", f"tsne_{data_set_name}_{generator_name}.png")
    plt.savefig(save_path, bbox_inches="tight")
    
    plt.close() # Close the figure to free up memory

# Iterate over all combinations of datasets and generators
for data_set_number, data_set_name in data_set_mapping.items():
    for generator_number, generator_name in generator_mapping.items():
        file_name = f"{data_set_name}_{generator_name}_emb.jsonl"
        file_path = os.path.join(folder_name, file_name)

        if os.path.exists(file_path):  # Check if the file exists
            process_and_visualize(file_path, data_set_name, generator_name)