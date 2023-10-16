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

# Ask user to select datasets
print("Available datasets:")
for num, dataset in data_set_mapping.items():
    print(f"{num}. {dataset}")

chosen_datasets_nums = input("Select one or more datasets by number (comma-separated, e.g., '1,3,4'): ").split(',')
chosen_datasets = [data_set_mapping[int(num)] for num in chosen_datasets_nums]

# Ask user to select generators
print("\nAvailable generators:")
for num, generator in generator_mapping.items():
    print(f"{num}. {generator}")

chosen_generators_nums = input("Select one or more generators by number (comma-separated, e.g., '2,5'): ").split(',')
chosen_generators = [generator_mapping[int(num)] for num in chosen_generators_nums]

def process_and_plot_datasets_and_generators(chosen_datasets, chosen_generators):
    embeddings = []
    labels = []
    processed_human_embeddings = set()

    for dataset_name in chosen_datasets:
        for generator_name in chosen_generators:
            file_name = f"{dataset_name}_{generator_name}_emb.jsonl"
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
                        labels.append(f"Machine-{dataset_name}-{generator_name}")

    # Convert list to numpy array
    X = np.array(embeddings)

    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plotting
    plt.figure(figsize=(12, 8))
    for unique_label in set(labels):
        mask = np.array(labels) == unique_label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=unique_label)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE visualization')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    # Ensure the 'figures' directory exists and save the figure there
    if not os.path.exists("figures"):
        os.makedirs("figures")

    save_name = f"tsne_datasets_{'_'.join(chosen_datasets)}_generators_{'_'.join(chosen_generators)}.png"
    save_path = os.path.join("figures", save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to {save_path}")

process_and_plot_datasets_and_generators(chosen_datasets, chosen_generators)