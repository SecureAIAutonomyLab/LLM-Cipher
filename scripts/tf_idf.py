"""
Generates TF_IDF plots
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


file_paths = [
     '../data/arxiv_bloomz.jsonl',
    '../data/arxiv_chatGPT.jsonl',
    '../data/arxiv_cohere.jsonl',
    '../data/arxiv_davinci.jsonl',
    '../data/arxiv_dolly.jsonl',
    '../data/arxiv_flant5.jsonl',
    '../data/peerread_bloomz.jsonl',
    '../data/peerread_cohere.jsonl',
    '../data/peerread_davinci.jsonl',
    '../data/peerread_dolly.jsonl',
    '../data/peerread_llama.jsonl',
    '../data/reddit_bloomz.jsonl',
    '../data/reddit_chatGPT.jsonl',
    '../data/reddit_dolly.jsonl',
    '../data/reddit_flant5.jsonl',
    '../data/wikihow_chatGPT.jsonl',
    '../data/wikihow_cohere.jsonl',
    '../data/wikihow_davinci.jsonl',
    '../data/wikihow_dolly2.jsonl',
    '../data/wikipedia_chatgpt.jsonl',
    '../data/wikipedia_cohere.jsonl',
    '../data/wikipedia_davinci.jsonl',
    '../data/wikipedia_dolly.jsonl',   
]


def get_top_features(tfidf_matrix, vectorizer, top_n=10):
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = np.argsort(tfidf_matrix.sum(axis=0).tolist()[0])[-top_n:][::-1]
    top_n_features = [feature_array[i] for i in tfidf_sorting]
    top_n_values = tfidf_matrix.sum(axis=0).tolist()[0]
    top_values = [top_n_values[i] for i in tfidf_sorting]
    return top_n_features, top_values


def plot_top_features(names_human, values_human, names_machine, values_machine, file_suffix):
    fig, axs = plt.subplots(nrows=2, figsize=(12, 12))
    
    sns.barplot(x=names_human, y=values_human, ax=axs[0])
    axs[0].set_title(f'Human Top TF-IDF Features')
    
    sns.barplot(x=names_machine, y=values_machine, ax=axs[1])
    axs[1].set_title(f'Machine Top TF-IDF Features')
    
    fig.suptitle(file_suffix.replace('_', ' ').title())
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{file_suffix}_combined_top_features.png')


def extract_model_from_path(file_path):
    """Extract model name from the file path."""
    return file_path.split('/')[-1].split('_')[1].lower()[:-6]  # Convert to lowercase for consistent comparison


def get_files_for_model(model_name):
    """Get all file paths that match the model name."""
    return [path for path in file_paths if extract_model_from_path(path) == model_name.lower()]  # Compare using lowercase


def main(model_name):
    """Main function to process all files for the given model name."""
    files = get_files_for_model(model_name)
    print(files)
    
    # Aggregating texts across files
    all_human_texts = []
    all_machine_texts = []

    for file_path in files:
        print(f'starting: {file_path}')
        
        with open(file_path, 'r') as file:
            for line in file:
                if len(line) < 10:
                    continue
                row = json.loads(line)

                key_pairs = [
                    ('human_text', 'machine_text'),
                    ('text', 'machine_text'),
                    ('abstract', 'machine_abstract'),
                    ('human_reviews', 'bloom_reviews')
                ]

                for human_key, machine_key in key_pairs:
                    if human_key in row and machine_key in row and row[human_key] and row[machine_key]:
                        human_text = row[human_key]
                        machine_text = row[machine_key]

                        if isinstance(human_text, list):
                            human_text = human_text[0]

                        if isinstance(machine_text, list):
                            machine_text = machine_text[0]

                        all_human_texts.append(human_text[:1000])
                        all_machine_texts.append(machine_text[:1000])
                        break
                
    # Applying TF-IDF on the aggregated texts
    vectorizer_human = TfidfVectorizer(stop_words='english')
    vectorizer_machine = TfidfVectorizer(stop_words='english')

    human_X = vectorizer_human.fit_transform(all_human_texts)
    machine_X = vectorizer_machine.fit_transform(all_machine_texts)

    # Getting top features
    human_features, human_values = get_top_features(human_X, vectorizer_human)
    machine_features, machine_values = get_top_features(machine_X, vectorizer_machine)

    # Plotting combined top features
    plot_top_features(human_features, human_values, machine_features, machine_values, model_name)

# Here you can specify the model name and run the main function
if __name__ == "__main__":
    chosen_model = "cohere"  # Example model name
    main(chosen_model)