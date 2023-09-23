import os
import json
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


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

def load_data_from_file(file_path):
    embeddings = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            embeddings.append(data['human_text_embedding'])
            labels.append(0)  # 0 for human
            embeddings.append(data['machine_text_embedding'])
            labels.append(1)  # 1 for machine
                
    return embeddings, labels

def write_statistics_to_file(filename, report_str):
    with open(filename, "a") as file:  # Using 'a' to append to the file
        file.write(report_str)

folder_name = "embedding_data"
results_folder = "knn_results"

# Ensure the 'knn_results' directory exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


results_filename = os.path.join(results_folder, "knn_individual_results.txt")

for data_set_name in data_set_mapping.values():
    for generator_name in generator_mapping.values():
        file_name = f"{data_set_name}_{generator_name}_emb.jsonl"
        file_path = os.path.join(folder_name, file_name)
        
        if os.path.exists(file_path):  
            embeddings, labels = load_data_from_file(file_path)
            X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

            # Training KNN
            knn = KNeighborsClassifier(n_neighbors=3)  # Adjust the number of neighbors as required
            knn.fit(X_train, y_train)

            # Evaluating on test data
            test_predictions = knn.predict(X_test)
            report = classification_report(y_test, test_predictions, target_names=['human', 'machine'], output_dict=True)
            report_str = f"Results for {file_name}:\n"
            report_str += f"Training samples: {len(X_train)}\n"
            report_str += f"Testing samples: {len(X_test)}\n"
            report_str += f"Accuracy: {report['accuracy']:.4f}\n"
            report_str += f"Human Precision: {report['human']['precision']:.4f}\n"
            report_str += f"Human Recall: {report['human']['recall']:.4f}\n"
            report_str += f"Human F1: {report['human']['f1-score']:.4f}\n"
            report_str += f"Machine Precision: {report['machine']['precision']:.4f}\n"
            report_str += f"Machine Recall: {report['machine']['recall']:.4f}\n"
            report_str += f"Machine F1: {report['machine']['f1-score']:.4f}\n"
            report_str += "----------------------\n"
            
            write_statistics_to_file(results_filename, report_str)
            print(f"Finished processing {file_name}")

print(f"All results saved to {results_filename}")