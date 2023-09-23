import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
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

embeddings = []
labels = []

processed_human_embeddings = set()  # to keep track of human embeddings

# Iterate over all combinations of datasets and generators
for data_set_number, data_set_name in data_set_mapping.items():
    for generator_number, generator_name in generator_mapping.items():
        file_name = f"{data_set_name}_{generator_name}_emb.jsonl"
        file_path = os.path.join(folder_name, file_name)
        
        if os.path.exists(file_path):  # Check if the file exists
            with open(file_path, "r") as file:
                for line in tqdm(file, desc=f"Processing {file_name}", leave=False):
                    data = json.loads(line)
                    human_emb = tuple(data['human_text_embedding'])  # convert list to tuple to make it hashable

                    if human_emb not in processed_human_embeddings:
                        embeddings.append(data['human_text_embedding'])
                        labels.append(f"Human-{data_set_name}")
                        processed_human_embeddings.add(human_emb)

                    embeddings.append(data['machine_text_embedding'])
                    labels.append(f"Machine-{data_set_name}-{generator_name}")

# Convert list to numpy array
X = np.array(embeddings)
X = normalize(X)

X = X.astype(np.float32)
print(np.min(X))
similarity_matrix = cosine_similarity(X)

# Ask if user wants to use cosine distance
use_cosine = input("Do you want to use cosine distance? (yes/no): ").strip().lower() == 'yes'

# If cosine distance is chosen, calculate the cosine distance matrix
if use_cosine:
    similarity_matrix = cosine_similarity(X)
    X = 1.0 - similarity_matrix  # Convert similarity to distance
    metric = 'precomputed'
else:
    metric = 'euclidean'

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, metric=metric, init="random")
X_tsne = tsne.fit_transform(X)

# Plotting
plt.figure(figsize=(12, 8))
for idx, label in enumerate(set(labels)):
    mask = np.array(labels) == label
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label)

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE visualization of embeddings')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
plt.grid(True)
plt.tight_layout()

# Ensure the 'figures' directory exists and save the figure there
if not os.path.exists("figures"):
    os.makedirs("figures")

filename = "tsne_visualization"
if use_cosine:
    filename += "_cosine"
save_path = os.path.join("figures", f"{filename}.png")
plt.savefig(save_path, bbox_inches="tight")

# Display the figure
plt.show()

print(f"Figure saved to {save_path}")