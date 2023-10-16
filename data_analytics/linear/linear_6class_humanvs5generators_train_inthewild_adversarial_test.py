import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


# Set up folders and classes
results_folder = "../../results/linear_results"
classes = ['bloomz', 'chatgpt', 'cohere', 'davinci', 'dolly', 'human']


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

# Training loop
best_val_accuracy = 0.0  # Initialize best validation accuracy
best_model_path = os.path.join(results_folder, 'best_model_humanvs5generators.pth')  # Path to save the best model


# Load the best model for evaluation and further testing
model.load_state_dict(torch.load(best_model_path))




    

# Set up the in-the-wild test folder
in_the_wild_folder = "../../data/embedding_data_T5_adversarial_npy"
#in_the_wild_folder = "../../data/embedding_data_T5_inthewild_npy"

# Load and preprocess in-the-wild data
def load_in_the_wild_data(folder_name, classes):
    dataset_stats = {}
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        embeddings = np.load(file_path)[:200]
        X_wild = torch.tensor(embeddings, dtype=torch.float32).to(device)  # Ensure wild data is on the correct device
        
        # Getting the raw logits from the model
        wild_outputs = model(X_wild)
        
        # Get the predicted class indices
        _, predicted = torch.max(wild_outputs, 1)
        
        # Computing average logits, standard deviation for the predicted classes
        avg_logits = {}
        std_logits = {}
        for class_idx, class_name in enumerate(classes):
            class_logits = wild_outputs[predicted == class_idx]
            if class_logits.size(0) != 0:
                avg_logits[class_name] = torch.mean(class_logits, dim=0)[class_idx].item()
                std_logits[class_name] = torch.std(class_logits, dim=0, unbiased=True)[class_idx].item()
            else:
                avg_logits[class_name] = float('nan')  # or set to a default value
                std_logits[class_name] = float('nan')  # or set to a default value
        
        # Counting the number of predictions for each class
        predicted_counts = {class_name: (predicted == class_idx).sum().item() for class_idx, class_name in enumerate(classes)}
        
        # Storing counts, average logits, and standard deviation
        dataset_stats[file_name] = {'counts': predicted_counts, 'avg_logits': avg_logits, 'std_logits': std_logits}
    
    return dataset_stats

# Get statistics for in-the-wild datasets
wild_stats = load_in_the_wild_data(in_the_wild_folder, classes)

# Save the in-the-wild dataset statistics to a .txt file
with open(os.path.join(results_folder, 'in_the_wild_adversarial_stats_humanvs5generators.txt'), 'w') as f:
    for dataset, stats in wild_stats.items():
        f.write(f"Statistics for {dataset}:\n")
        for class_name in classes:
            count = stats['counts'].get(class_name, 0)
            avg_logit = stats['avg_logits'].get(class_name, float('nan'))
            std_logit = stats['std_logits'].get(class_name, float('nan'))
            f.write(f"{class_name}: {count}, Average Logit: {avg_logit}, Std Dev Logit: {std_logit}\n")
        f.write("-------------------------------\n")
