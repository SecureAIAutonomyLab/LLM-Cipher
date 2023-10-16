import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


# Set up folders and classes
results_folder = "../../results/linear_results"
classes = ['human', 'machine']
best_model_path = os.path.join(results_folder, 'best_model_linear_binary_2.pth')

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




# Define Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 2048)  
        self.fc3 = nn.Linear(2048, 1024)  
        self.fc4 = nn.Linear(1024, 512)   
        self.fc5 = nn.Linear(512, 256)   
        self.fc6 = nn.Linear(256, 6)      
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


# Initialize and train the neural network
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in model: {total_params}')

      

# Load the best model for evaluation and further testing
model.load_state_dict(torch.load(best_model_path))


    

# Set up the in-the-wild test folder
in_the_wild_folder = "../../data/embedding_data_T5_inthewild_npy"
dataset_names = ['ai_writer', 'ArticleForge', 'kafkai', 'reddit_bot']
results_file = os.path.join(results_folder, 'inthewild_binary_stats_2.txt')

# Dictionary to store datasets
datasets = {name: {'data': [], 'labels': []} for name in dataset_names}
for file in os.listdir(in_the_wild_folder):
    for dataset in dataset_names:
        if dataset in file:
            data = np.load(os.path.join(in_the_wild_folder, file))[:200]
            label = 0 if 'human' in file else 1  # 0 for human, 1 for machine
            datasets[dataset]['data'].extend(data)
            datasets[dataset]['labels'].extend([label] * len(data))

# Evaluate each dataset
with open(results_file, 'w') as file:
    for dataset, value in datasets.items():
        data = np.array(value['data'])
        true_labels = np.array(value['labels'])

        # Convert to tensor and make predictions (Ensure your model can handle the input)
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        with torch.no_grad():
            model.eval()
            outputs = model(data_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()

        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None, labels=[0, 1], zero_division=1)

        # Write results to file
        file.write(f"Results for {dataset}:\n")
        file.write(f"Accuracy: {accuracy:.3f}\n")
        file.write(f"Human Precision: {precision[0]:.3f}, Human Recall: {recall[0]:.3f}, Human F1: {f1[0]:.3f}\n")
        file.write(f"Machine Precision: {precision[1]:.3f}, Machine Recall: {recall[1]:.3f}, Machine F1: {f1[1]:.3f}\n")
        file.write("\n")

print(f"Results written to {results_file}")