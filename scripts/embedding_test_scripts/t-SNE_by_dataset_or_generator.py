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

def process_embeddings_and_plot(mode, constant_name):
    embeddings = []
    labels = []
    processed_human_embeddings = set()  # to keep track of human embeddings

    if mode == "dataset":
        chosen_dataset = constant_name
        for generator_name in generator_mapping.values():
            file_name = f"{chosen_dataset}_{generator_name}_emb.jsonl"
            file_path = os.path.join(folder_name, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    for line in tqdm(file, desc=f"Processing {file_name}", leave=False):
                        data = json.loads(line)
                        human_emb = tuple(data['human_text_embedding'])
                        
                        if human_emb not in processed_human_embeddings:
                            embeddings.append(data['human_text_embedding'])
                            labels.append(f"Human-{chosen_dataset}")
                            processed_human_embeddings.add(human_emb)

                        embeddings.append(data['machine_text_embedding'])
                        labels.append(f"Machine-{chosen_dataset}-{generator_name}")

    elif mode == "generator":
        chosen_generator = constant_name
        for dataset_name in data_set_mapping.values():
            file_name = f"{dataset_name}_{chosen_generator}_emb.jsonl"
            file_path = os.path.join(folder_name, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    for line in tqdm(file, desc=f"Processing {file_name}", leave=False):
                        data = json.loads(line)
                        human_emb = tuple(data['human_text_embedding'])
                        
                        if human_emb not in processed_human_embeddings:
                            embeddings.append(data['human_text_embedding'])
                            labels.append(f"Human-{dataset_name}")
                            processed_human_embeddings.add(human_emb)

                        embeddings.append(data['machine_text_embedding'])
                        labels.append(f"Machine-{dataset_name}-{chosen_generator}")

    # Convert list to numpy array
    X = np.array(embeddings)

    # 2. t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plotting
    plt.figure(figsize=(12, 8))
    for unique_label in set(labels):
        mask = np.array(labels) == unique_label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=unique_label)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f't-SNE visualization ({constant_name} constant)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    # Ensure the 'figures' directory exists and save the figure there
    if not os.path.exists("figures"):
        os.makedirs("figures")

    save_path = os.path.join("figures", f"tsne_{mode}_{constant_name}_constant.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()  # Close the current figure to save memory

    print(f"Figure saved to {save_path}")

# Run t-SNE for all datasets and all generators
for dataset_name in data_set_mapping.values():
    process_embeddings_and_plot("dataset", dataset_name)

for generator_name in generator_mapping.values():
    process_embeddings_and_plot("generator", generator_name)