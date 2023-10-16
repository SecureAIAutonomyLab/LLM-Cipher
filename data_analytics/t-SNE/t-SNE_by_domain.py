import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

perturbation = 'original'

folder_name = "../../data/embedding_data_T5_npy/"
results_folder = "../../results/tsne_results"

data_set_mapping = {
    1: "arxiv",
    2: "peerread",
    3: "reddit",
    4: "wikihow",
    5: "wikipedia"
}

generator_mapping = {
    1: "bloomz",
    2: "chatgpt",
    3: "cohere",
    4: "davinci",
    5: "dolly"
}

color_mapping = {
    "Human": "#1f77b4",  # muted blue
    "bloomz": "#ff7f0e",  # muted orange
    "chatgpt": "#2ca02c",  # muted green
    "cohere": "#17becf",   # muted teal
    "davinci": "#9467bd",  # muted purple
    "dolly": "#8c564b"   # muted brown
}

perturbation_mapping = {
    "original": "original",
    "contrastive": "contrastive"    
}

def load_embeddings(generator_name, data_set_name, perturbation_type, entity_type):
    file_name = f"{generator_name}_{data_set_name}_{entity_type}_{perturbation_type}_cleaned.npy"
    file_path = os.path.join(folder_name, file_name)
    if os.path.exists(file_path):
        return np.load(file_path)
    return None

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

for data_set_name in data_set_mapping.values():
    if perturbation in perturbation_mapping:
        all_embeddings = None
        labels = []

        plt.figure(figsize=(12, 8))
        
        for generator_name in generator_mapping.values():
            human_embeddings = load_embeddings(generator_name, data_set_name, perturbation, "human")
            machine_embeddings = load_embeddings(generator_name, data_set_name, perturbation, "machine")

            if human_embeddings is not None:
                if all_embeddings is None:
                    all_embeddings = human_embeddings
                else:
                    all_embeddings = np.concatenate((all_embeddings, human_embeddings), axis=0)
                labels.extend(["Human"] * len(human_embeddings))

            if machine_embeddings is not None:
                if all_embeddings is None:
                    all_embeddings = machine_embeddings
                else:
                    all_embeddings = np.concatenate((all_embeddings, machine_embeddings), axis=0)
                labels.extend([generator_name] * len(machine_embeddings))

        if all_embeddings is not None:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(all_embeddings)

            for label in set(labels):
                plt.scatter(tsne_results[np.array(labels) == label, 0], 
                            tsne_results[np.array(labels) == label, 1], 
                            label=label, 
                            color=color_mapping[label]) 

        #plt.title(f"t-SNE Results for {data_set_name} with {perturbation} Perturbation")
        #plt.legend()
        plt.axis('off')
        plt.savefig(os.path.join(results_folder, f"tsne_combined_human_and_generators_{data_set_name}_{perturbation}.png"))
        plt.close()

        print(f"Finished processing {data_set_name} with {perturbation} perturbation")
    else:
        print(f"Perturbation type {perturbation} is not recognized. Skipping...")

print("All results saved in", results_folder)