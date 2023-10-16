import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score

# Set up your mappings and folders
train_folder_name = "../../data/embedding_data_T5_npy"
test_folder_name = "../../data/embedding_data_T5_adversarial_npy"
results_folder = "../../results/knn_results"

def load_embeddings(folder_name, dataset_name, entity_type, test_limit=None):
    embeddings_list = []
    labels_list = []
    for file_name in os.listdir(folder_name):
        if dataset_name in file_name and entity_type in file_name:
            file_path = os.path.join(folder_name, file_name)
            embeddings = np.load(file_path)
            if test_limit and 'test' in folder_name:  # If it's a test folder and limit is specified, take only limited samples
                embeddings = embeddings[:test_limit]
            labels = np.zeros(embeddings.shape[0]) if entity_type == 'human' else np.ones(embeddings.shape[0])
            
            # Print the number of samples in the current .npy file
            print(f"Number of samples in {file_name}: {embeddings.shape[0]}")
            
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    
    if not embeddings_list:
        return None, None
    
    return np.concatenate(embeddings_list, axis=0), np.concatenate(labels_list, axis=0)

def write_statistics_to_file(filename, report_str):
    with open(filename, "a") as file:
        file.write(report_str)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Load and preprocess training data
X_train_human, y_train_human = load_embeddings(train_folder_name, '', 'human')
X_train_machine, y_train_machine = load_embeddings(train_folder_name, '', 'machine')
X_train = np.concatenate((X_train_human, X_train_machine), axis=0)
y_train = np.concatenate((y_train_human, y_train_machine), axis=0)

# Print the number of samples in the training human and machine sets
print(f"Number of samples in training human set: {X_train_human.shape[0]}")
print(f"Number of samples in training machine set: {X_train_machine.shape[0]}")

# Train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_classifier.fit(X_train, y_train)

# List of datasets to evaluate
datasets = ['ai_writer', 'ArticleForge', 'kafkai', 'reddit_bot_gpt3']

# Evaluate on each dataset
for dataset in datasets:
    X_test_human, y_test_human = load_embeddings(test_folder_name, dataset, 'human', 200)
    X_test_machine, y_test_machine = load_embeddings(test_folder_name, dataset, 'machine', 200)
    if X_test_human is None or X_test_machine is None:
        print(f"Skipping {dataset} due to missing data.")
        continue
    X_test = np.concatenate((X_test_human, X_test_machine), axis=0)
    y_test = np.concatenate((y_test_human, y_test_machine), axis=0)

    # Predict and calculate metrics
    y_pred = knn_classifier.predict(X_test)
    accuracy = knn_classifier.score(X_test, y_test)
    precision, recall, fscore, support = score(y_test, y_pred, labels=[0, 1])

    # Write results to file
    results_str = f"Results for {dataset}:\n"
    results_str += f"Accuracy: {accuracy:.3f}\n"
    results_str += f"Human Precision: {precision[0]:.3f}, Human Recall: {recall[0]:.3f}, Human F1: {fscore[0]:.3f}\n"
    results_str += f"Machine Precision: {precision[1]:.3f}, Machine Recall: {recall[1]:.3f}, Machine F1: {fscore[1]:.3f}\n"
    results_str += "----------------------------------\n"
    result_file_name = f"knn_results_{dataset}_adversarial.txt"
    write_statistics_to_file(os.path.join(results_folder, result_file_name), results_str)

    print(f"Evaluation completed for {dataset}. Check the results folder for the detailed report.")
