import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score

# Set up your mappings and folders
folder_name = "../../data/embedding_data_contrastive_npy/"
results_folder = "../../results/knn_contrasti_results"

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

# Set up your argument parser
parser = argparse.ArgumentParser(description='Run KNN with specified parameters.')
parser.add_argument('--n_neighbors', type=int, default=1, help='Value for n_neighbors')
parser.add_argument('--metric', default='euclidean', help='Value for metric')
parser.add_argument('--generator', choices=generator_mapping.values(), required=True, help='The generator to use')
parser.add_argument('--perturbation', type=str, default='original', help='Type of perturbation')
args = parser.parse_args()

n_neighbors = args.n_neighbors
metric = args.metric
constant_generator = args.generator
perturbation = args.perturbation

# Define functions to load embeddings and write results
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

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

def load_and_preprocess_data(generator_name, data_set_name, perturbation_type):
    X_human, y_human = load_embeddings(generator_name, data_set_name, perturbation_type, 'human')
    X_machine, y_machine = load_embeddings(generator_name, data_set_name, perturbation_type, 'machine')

    if X_human is None or X_machine is None:
        return None, None

    X = np.concatenate((X_human, X_machine), axis=0)
    y = np.concatenate((y_human, y_machine), axis=0)
    
    return X, y

# Outer loop: Loop through each dataset for training

for train_dataset in data_set_mapping.values():
    X_train_full, y_train_full = load_and_preprocess_data(constant_generator, train_dataset, perturbation)

    if X_train_full is None:
        continue

    X_train, X_temp, y_train, y_temp = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn_classifier.fit(X_train, y_train)

    for test_dataset in data_set_mapping.values():
        X_test_full, y_test_full = load_and_preprocess_data(constant_generator, test_dataset, perturbation)

        if X_test_full is None:
            continue

        X_train_unused, X_temp, y_train_unused, y_temp = train_test_split(X_test_full, y_test_full, test_size=0.2, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        for split_name, X_split, y_split in zip(['Test', 'Validation'], [X_test, X_val], [y_test, y_val]):
            y_pred = knn_classifier.predict(X_split)
            accuracy = knn_classifier.score(X_split, y_split)
            precision, recall, fscore, support = score(y_split, y_pred, labels=[0, 1])

            results_str = f"Train on {train_dataset}, {split_name} on {test_dataset}\n"
            results_str += f"Samples in {split_name} set: {len(y_split)}\n"
            results_str += f"Accuracy: {accuracy:.2f}\n"
            results_str += f"Human Precision: {precision[0]:.2f}, Human Recall: {recall[0]:.2f}, Human F1: {fscore[0]:.2f}\n"
            results_str += f"Machine Precision: {precision[1]:.2f}, Machine Recall: {recall[1]:.2f}, Machine F1: {fscore[1]:.2f}\n"
            results_str += "----------------------------------\n"  # Separator for readability

            result_file_name = f"cross_domain_knn_{constant_generator}_{n_neighbors}_{metric}_{perturbation}.txt"
            write_statistics_to_file(os.path.join(results_folder, result_file_name), results_str)
            
            print(f"Trained on: {train_dataset}. Evaluated on: {test_dataset}. Split: {split_name}. Constant Generator: {constant_generator}. Number of Neighbors: {n_neighbors}. Metric: {metric}. Perturbation: {perturbation}.")
