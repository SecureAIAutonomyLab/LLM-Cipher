import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
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
def load_in_the_wild_data(folder_name):
    domain_names = ['ai_writer', 'ArticleForge', 'kafkai', 'reddit_bot']
    
    domain_stats = {name: {'true': [], 'pred': []} for name in domain_names}
    
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        embeddings = np.load(file_path)[:200]
        X_wild = torch.tensor(embeddings, dtype=torch.float32).to(device)  # Ensure wild data is on the correct device
        
        # Getting the raw logits from the model
        wild_outputs = model(X_wild)
        
        # Get the predicted class indices
        _, predicted = torch.max(wild_outputs, 1)
        predicted_human_or_machine = ["human" if classes[p] == "human" else "machine" for p in predicted.cpu().numpy()]
        
        for domain in domain_names:
            if domain in file_name:
                if 'human' in file_name:
                    domain_stats[domain]['true'].extend(["human"] * len(predicted_human_or_machine))
                else:
                    domain_stats[domain]['true'].extend(["machine"] * len(predicted_human_or_machine))
                domain_stats[domain]['pred'].extend(predicted_human_or_machine)
                break
    
    # Now that we've collected all true and predicted labels, we'll compute the metrics
    for domain, data in domain_stats.items():
        true_labels = data['true']
        pred_labels = data['pred']
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=["human", "machine"], zero_division=1)
        
        domain_stats[domain]['accuracy'] = accuracy
        domain_stats[domain]['precision'] = precision
        domain_stats[domain]['recall'] = recall
        domain_stats[domain]['f1'] = f1
    
    return domain_stats

wild_stats = load_in_the_wild_data(in_the_wild_folder)

# Now let's print/save these stats
with open(os.path.join(results_folder, 'in_the_wild_adversarial_stats_humanvs5generators.txt'), 'w') as f:
    for domain, stats in wild_stats.items():
        f.write(f"Results for {domain}:\n")
        f.write(f"Accuracy: {stats['accuracy']:.3f}\n")
        f.write(f"Human Precision: {stats['precision'][0]:.3f}, Human Recall: {stats['recall'][0]:.3f}, Human F1: {stats['f1'][0]:.3f}\n")
        f.write(f"Machine Precision: {stats['precision'][1]:.3f}, Machine Recall: {stats['recall'][1]:.3f}, Machine F1: {stats['f1'][1]:.3f}\n")
        f.write("\n")