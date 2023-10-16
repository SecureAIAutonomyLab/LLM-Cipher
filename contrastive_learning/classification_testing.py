import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from typing import List
import json
from contrastive_utils.datasets import InverseClassificationDataset, InTheWild, Adversarial
from contrastive_utils.models import SimpleNN


def test_model(checkpoint_path: str, 
        root_dir: str, 
        include_filters: List[str], 
        num_classes: int, 
        device='cuda',
        save_dir=None
    ) -> None:

    results_save_name = checkpoint_path.split('/')[1]
    model = SimpleNN(num_classes=num_classes)  
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    dataset = InverseClassificationDataset(root_dir, include_filters)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Only collapse to binary if num_classes is 2, else let it be multi-class
    if num_classes == 2 or include_filters[0] != '':
        all_preds = [0 if pred == 0 else 1 for pred in all_preds]
        all_labels = [0 if label == 0 else 1 for label in all_labels]
        target_names = ["Human", "Machine"]
    else:
        target_names = ["Human", "Bloomz", "ChatGPT", "Cohere", "Davinci", "Dolly"]

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n\nConfusion Matrix:")
    print(conf_matrix)

    print("\nMetrics:")
    for label in target_names:
        print(f"{label} - Precision: {report[label]['precision']}")
        print(f"{label} - Recall: {report[label]['recall']}")
        print(f"{label} - F1: {report[label]['f1-score']}")
        print(f"{label} - Support: {report[label]['support']}")

    results = {
        'Accuracy': accuracy,
        'Metrics': report,
        'Confusion Matrix': conf_matrix.tolist()
    }
    save_metrics_to_json(results, f'{save_dir}{results_save_name}.json')


def test_model_inthewild(checkpoint_path: str, 
        root_dir: str, 
        num_classes: int, 
        device='cuda',
        save_dir=None,
    ) -> None:

    results_save_name = checkpoint_path.split('/')[1]
    model = SimpleNN(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    dataset = InTheWild(root_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Check if the model is binary or multi-class
    if num_classes == 2:
        target_names = ["Human", "Machine"]
    else:
        # If multi-class, treat any non-zero prediction as "Machine"
        all_preds = [1 if pred > 0 else 0 for pred in all_preds]
        target_names = ["Human", "Machine"]

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n\nConfusion Matrix:")
    print(conf_matrix)

    print("\nMetrics:")
    for label in target_names:
        print(f"{label} - Precision: {report[label]['precision']}")
        print(f"{label} - Recall: {report[label]['recall']}")
        print(f"{label} - F1: {report[label]['f1-score']}")
        print(f"{label} - Support: {report[label]['support']}")

    results = {
        'Accuracy': accuracy,
        'Metrics': report,
        'Confusion Matrix': conf_matrix.tolist()
    }
    save_metrics_to_json(results, f'{save_dir}{results_save_name}.json')


def test_model_adversarial(checkpoint_path: str, 
        root_dir: str, 
        num_classes: int, 
        device='cuda',
        save_dir=None,
    ) -> None:

    results_save_name = checkpoint_path.split('/')[1]
    model = SimpleNN(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    dataset = Adversarial(root_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if num_classes == 2:
        target_names = ["Human", "Machine"]
    else:
        all_preds = [1 if pred > 0 else 0 for pred in all_preds]
        target_names = ["Human", "Machine"]

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n\nConfusion Matrix:")
    print(conf_matrix)

    print("\nMetrics:")
    for label in target_names:
        print(f"{label} - Precision: {report[label]['precision']}")
        print(f"{label} - Recall: {report[label]['recall']}")
        print(f"{label} - F1: {report[label]['f1-score']}")
        print(f"{label} - Support: {report[label]['support']}")

    results = {
        'Accuracy': accuracy,
        'Metrics': report,
        'Confusion Matrix': conf_matrix.tolist()
    }
    save_metrics_to_json(results, f'{save_dir}{results_save_name}.json')


def save_metrics_to_json(metrics, save_path):
    with open(save_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)


experiments = [
    ('allm4_2class', 2, ''),
    ('allm4_6class', 6, ''),
    ('chatgpt_heldout_5class', 5, 'chatgpt'),
    ('bloomz_heldout_5class', 5, 'bloomz'),
    ('cohere_heldout_5class', 5, 'cohere'),
    ('davinci_heldout_5class', 5, 'davinci'),
    ('dolly_heldout_5class', 5, 'dolly'),
    ('peerread_heldout_6class', 6, 'peerread'),
    ('reddit_heldout_6class', 6, 'reddit'),
    ('wikihow_heldout_6class', 6, 'wikihow'),
    ('wikipedia_heldout_6class', 6, 'wikipedia'),
    ('arxiv_heldout_6class', 6, 'arxiv'),
    ('chatgpt_heldout_2class', 2, 'chatgpt'),
    ('bloomz_heldout_2class', 2, 'bloomz'),
    ('cohere_heldout_2class', 2, 'cohere'),
    ('davinci_heldout_2class', 2, 'davinci'),
    ('dolly_heldout_2class', 2, 'dolly'),
    ('peerread_heldout_2class', 2, 'peerread'),
    ('reddit_heldout_2class', 2, 'reddit'),
    ('wikihow_heldout_2class', 2, 'wikihow'),
    ('wikipedia_heldout_2class', 2, 'wikipedia'),
    ('arxiv_heldout_2class', 2, 'arxiv'),
]


base_dir = 'classification_results/'
model_dir = 'results_0/'
embedding_data_path = '../data/embedding_data_T5_npy/'

for model_name, num_classes, keyword in experiments:
    model_path = model_dir + model_name + '/contrastive_model_best.pt' 
    test_model(model_path, embedding_data_path, [keyword], num_classes=num_classes, save_dir=base_dir)

experiments = [
    ('allm4_2class', 2),
    ('allm4_6class', 6),
]

base_dir = 'classification_results_inthewild/'
model_dir = 'results_0/'
embedding_data_path = '../data/embedding_data_T5_inthewild_npy/'

for model_name, num_classes in experiments:
    model_path = model_dir + model_name + '/contrastive_model_best.pt' 
    test_model_inthewild(model_path, embedding_data_path, num_classes=num_classes, save_dir=base_dir)

experiments = [
    ('allm4_2class', 2),
    ('allm4_6class', 6),
]

base_dir = 'classification_results_adversarial/'
model_dir = 'results_0/'
embedding_data_path = '../data/embedding_data_T5_adversarial_npy/'

for model_name, num_classes in experiments:
    model_path = model_dir + model_name + '/contrastive_model_best.pt' 
    test_model_adversarial(model_path, embedding_data_path, num_classes=num_classes, save_dir=base_dir)
