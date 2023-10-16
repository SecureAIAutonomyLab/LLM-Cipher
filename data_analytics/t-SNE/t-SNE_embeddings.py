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

perturbation_mapping = {
    "original": "original",
    "contrastive": "contrastive"    
}

color_mapping = {
    "Human": "#1f77b4",    # muted blue
    "Machine": "#d62728"   # muted red
}

def load_embeddings(generator_name, data_set_name, perturbation_type, entity_type):
    file_name = f"{generator_name}_{data_set_name}_{entity_type}_{perturbation_type}_cleaned.npy"
    file_path = os.path.join(folder_name, file_name)
    if os.path.exists(file_path):
        return np.load(file_path)
    return None

# Ensure the 'tsne_results' directory exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Main loop
for data_set_name in data_set_mapping.values():
    if perturbation in perturbation_mapping:
        for generator_name in generator_mapping.values():
            # Load human and machine embeddings related to a specific generator, dataset, and perturbation type
            human_embeddings = load_embeddings(generator_name, data_set_name, perturbation, "human")
            machine_embeddings = load_embeddings(generator_name, data_set_name, perturbation, "machine")
            
            if human_embeddings is not None and machine_embeddings is not None:
                embeddings = np.concatenate((human_embeddings, machine_embeddings), axis=0)
                labels = np.concatenate((np.zeros(len(human_embeddings)), np.ones(len(machine_embeddings))), axis=0)
                
                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                tsne_results = tsne.fit_transform(embeddings)
                
                # Plot t-SNE results
                plt.figure(figsize=(12, 8))
                for label in [0, 1]:
                    indices = np.where(labels == label)
                    entity_type = "Human" if label == 0 else "Machine"
                    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                                label=f"{label} - {entity_type}", 
                                color=color_mapping[entity_type])
                
#                 plt.figure(figsize=(12, 8))
#                 for label in [0, 1]:
#                     indices = np.where(labels == label)
#                     plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f"{label} - {'Human' if label == 0 else 'Machine'}")
                
                #plt.title(f"t-SNE Results for {generator_name} on {data_set_name} with {perturbation} Perturbation")
                #plt.legend()
                plt.axis('off')
                plt.savefig(os.path.join(results_folder, f"tsne_{generator_name}_{data_set_name}_{perturbation}.png"))
                plt.close()

                #print(f"Finished processing {generator_name} on {data_set_name} with {perturbation} perturbation")
                print(f"Finished processing {generator_name} on {data_set_name} with {perturbation} perturbation")

    else:
        print(f"Perturbation type {perturbation} is not recognized. Skipping...")

print("All results saved in", results_folder)
