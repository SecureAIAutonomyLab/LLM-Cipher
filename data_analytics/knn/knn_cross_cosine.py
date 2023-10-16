import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score

# Argument parser setup
parser = argparse.ArgumentParser(description='KNN classifier for held out data.')
parser.add_argument('--held_out', required=True, type=str, help='Held out value for filtering filenames.')
args = parser.parse_args()
held_out = args.held_out

# Set up your mappings and folders
folder_name = f"../../data/embedding_data_T5_npy/"
results_folder = "../../results/knn_results"
results_file = f"knn_results_held_out_{held_out}_cosine.txt"

# Load data
train_data, train_labels, test_data, test_labels = [], [], [], []

for file in os.listdir(folder_name):
    data = np.load(os.path.join(folder_name, file))
    
    # Classify based on the file name
    if 'human' in file:
        label = 1
    else:
        label = 0
    
    if held_out in file:
        test_data.extend(data)
        test_labels.extend([label]*len(data))
    else:
        train_data.extend(data)
        train_labels.extend([label]*len(data))

train_data, test_data = np.array(train_data), np.array(test_data)
train_labels, test_labels = np.array(train_labels), np.array(test_labels)

# Build KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(train_data, train_labels)

# Predict
preds = knn.predict(test_data)

# Metrics
accuracy = accuracy_score(test_labels, preds)
precision, recall, f1, _ = score(test_labels, preds, average=None, labels=[1, 0])
metrics = {
    "Accuracy": accuracy,
    "Human": {"Precision": precision[0], "Recall": recall[0], "F1": f1[0]},
    "Machine": {"Precision": precision[1], "Recall": recall[1], "F1": f1[1]}
}

# Save results
with open(os.path.join(results_folder, results_file), 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("\nClass Human:\n")
    f.write(f"Precision: {metrics['Human']['Precision']:.4f}\n")
    f.write(f"Recall: {metrics['Human']['Recall']:.4f}\n")
    f.write(f"F1 Score: {metrics['Human']['F1']:.4f}\n")
    
    f.write("\nClass Machine:\n")
    f.write(f"Precision: {metrics['Machine']['Precision']:.4f}\n")
    f.write(f"Recall: {metrics['Machine']['Recall']:.4f}\n")
    f.write(f"F1 Score: {metrics['Machine']['F1']:.4f}\n")

