import os
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

folder_name = "embedding_data"
results_folder = "knn_results"

# Make sure the directory exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

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

processed_human_embeddings = set()  # to keep track of human embeddings

def load_data_from_file(file_path):
    embeddings = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            human_emb = tuple(data['human_text_embedding'])

            if human_emb not in processed_human_embeddings:
                embeddings.append(data['human_text_embedding'])
                labels.append(0)
                processed_human_embeddings.add(human_emb)

            embeddings.append(data['machine_text_embedding'])
            labels.append(1)
                
    return embeddings, labels

def write_statistics_to_file(filename, report_str):
    with open(filename, "a") as file:  # Using 'a' to append to the file
        file.write(report_str)

for excluded_generator in generator_mapping.values():
    processed_human_embeddings.clear()  # Clear for each generator
    X_train, y_train = [], []
    for gen in generator_mapping.values():
        if gen != excluded_generator:
            for data_set in data_set_mapping.values():
                embeddings, labels = load_data_from_file(os.path.join(folder_name, f"{data_set}_{gen}_emb.jsonl"))
                X_train.extend(embeddings)
                y_train.extend(labels)

    processed_human_embeddings.clear()  # Clear before loading test data
    X_test, y_test = [], []
    for data_set in data_set_mapping.values():
        embeddings, labels = load_data_from_file(os.path.join(folder_name, f"{data_set}_{excluded_generator}_emb.jsonl"))
        X_test.extend(embeddings)
        y_test.extend(labels)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['human', 'machine'], output_dict=True)
    
    # Create a string with the results
    report_str = f"Results when excluding {excluded_generator}:\n"
    report_str += f"Accuracy: {report['accuracy']:.4f}\n"
    report_str += f"Human Precision: {report['human']['precision']:.4f}, Recall: {report['human']['recall']:.4f}, F1: {report['human']['f1-score']:.4f}\n"
    report_str += f"Machine Precision: {report['machine']['precision']:.4f}, Recall: {report['machine']['recall']:.4f}, F1: {report['machine']['f1-score']:.4f}\n"
    report_str += "----------------------\n"

    results_filename = os.path.join(results_folder, f"knn_exclude_{excluded_generator}_results.txt")
    write_statistics_to_file(results_filename, report_str)
    print(f"Finished processing excluding {excluded_generator}")

print("All results saved to knn_results directory.")
