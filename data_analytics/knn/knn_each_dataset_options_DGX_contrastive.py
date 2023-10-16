import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Run KNN with specified n_neighbors and metric.')
parser.add_argument('--n_neighbors', type=int, choices=[1, 3, 5, 7], default=1, help='Value for n_neighbors')
parser.add_argument('--metric', choices=['cosine', 'euclidean', 'chebyshev'], default='euclidean', help='Value for metric')
parser.add_argument('--perturbation', type=str, default='original', help='Type of perturbation')
args = parser.parse_args()

# Extract provided values
n_neighbors = args.n_neighbors
metric = args.metric
perturbation = args.perturbation

folder_name = "../../data/embedding_data_contrastive_npy/"
results_folder = "../../results/knn_contrastive_results"

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

def load_embeddings(generator_name, data_set_name, perturbation_type, entity_type):
    file_name = f"{generator_name}_{data_set_name}_{entity_type}_{perturbation_type}.npy"
    file_path = os.path.join(folder_name, file_name)
    if os.path.exists(file_path):
        embeddings = np.load(file_path)
        embeddings = embeddings[250:]  # Skip the first 100 rows
        labels = np.zeros(embeddings.shape[0]) if entity_type == 'human' else np.ones(embeddings.shape[0])
        return embeddings, labels
    return None, None



def write_statistics_to_file(filename, report_str):
    with open(filename, "a") as file:
        file.write(report_str)

# Ensure the 'knn_results' directory exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Main loop
for data_set_name in data_set_mapping.values():
    if perturbation in perturbation_mapping:
        for generator_name in generator_mapping.values():
            # Load human embeddings related to a specific generator, dataset, and perturbation type
            human_embeddings = load_embeddings(generator_name, data_set_name, perturbation, "human")
            machine_embeddings = load_embeddings(generator_name, data_set_name, perturbation, "machine")
            
            if human_embeddings is not None and machine_embeddings is not None:
                embeddings = np.concatenate((human_embeddings, machine_embeddings), axis=0)
                labels = np.concatenate((np.zeros(len(human_embeddings)), np.ones(len(machine_embeddings))), axis=0)
                
                # Create train-test-val splits
                train_indices, temp_indices = train_test_split(np.arange(len(embeddings)), test_size=0.2, random_state=42)
                test_indices, val_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
                
                datasets = {
                    "train": (embeddings[train_indices], labels[train_indices]),
                    "test": (embeddings[test_indices], labels[test_indices]),
                    "val": (embeddings[val_indices], labels[val_indices])
                }
                
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                knn.fit(*datasets["train"])
                
                # The following is the loop where you evaluate your model on test and val
                for set_name, (X, y) in datasets.items():
                    if set_name != "train":  # No need to evaluate on training data
                        predictions = knn.predict(X)
                        report = classification_report(y, predictions, target_names=['human', 'machine'], output_dict=True)
                        report_str = f"Results for {generator_name} on {data_set_name} {set_name} set with {perturbation} perturbation:\n"
                        report_str += f"Samples: {len(X)}\n"
                        report_str += f"Accuracy: {report['accuracy']:.4f}\n"
                        report_str += f"Human Precision: {report['human']['precision']:.4f}\n"
                        report_str += f"Human Recall: {report['human']['recall']:.4f}\n"
                        report_str += f"Human F1: {report['human']['f1-score']:.4f}\n"
                        report_str += f"Machine Precision: {report['machine']['precision']:.4f}\n"
                        report_str += f"Machine Recall: {report['machine']['recall']:.4f}\n"
                        report_str += f"Machine F1: {report['machine']['f1-score']:.4f}\n"
                        report_str += "----------------------\n"

                        results_filename = os.path.join(results_folder, f"knn_results_indistribution_{perturbation}_{set_name}_{n_neighbors}neighbors_{metric}.txt")
                        write_statistics_to_file(results_filename, report_str)

                        print(f"Finished processing {generator_name} on {data_set_name} {set_name} set with {perturbation} perturbation")

    else:
        print(f"Perturbation type {perturbation} is not recognized. Skipping...")

print("All results saved in", results_folder)
