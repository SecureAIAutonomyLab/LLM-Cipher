import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# Set up folders and classes
train_folder_name = "../../data/embedding_data_T5_npy"
results_folder = "../../results/linear_results"
classes = ['bloomz', 'chatgpt', 'cohere', 'davinci', 'dolly', 'human']

# Load and preprocess data
def load_and_preprocess_data(folder_name, classes):
    embeddings_list = []
    labels_list = []
    
    for class_idx, class_name in enumerate(classes):
        for file_name in os.listdir(folder_name):
            # Condition to check if file should be loaded into the current class
            if class_name == 'human' and class_name in file_name:
                load_condition = True
            elif class_name != 'human' and class_name in file_name and 'machine' in file_name:
                load_condition = True
            else:
                load_condition = False
            
            # Load file if condition is met
            if load_condition:
                file_path = os.path.join(folder_name, file_name)
                embeddings = np.load(file_path)
                
                labels = np.full(embeddings.shape[0], class_idx)
                embeddings_list.append(embeddings)
                labels_list.append(labels)
    
    X = np.concatenate(embeddings_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return train_test_split(X, y, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = load_and_preprocess_data(train_folder_name, classes)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

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

best_model_path = os.path.join(results_folder, 'best_model_humanvs5generators.pth') 
model = SimpleNN().to(device)
model.load_state_dict(torch.load(best_model_path))

# Evaluate the model
X_test, y_test = X_test.to(device), y_test.to(device)
test_outputs = model(X_test)
_, predicted = torch.max(test_outputs, 1)

# Move tensors to CPU and convert to NumPy before passing to classification_report
y_test_np = y_test.cpu().numpy()
predicted_np = predicted.cpu().numpy()

class_report = classification_report(y_test_np, predicted_np, target_names=classes, zero_division=1, digits=3)

# Save the classification report
with open(os.path.join(results_folder, 'classification_report_humanvs5generators_2.txt'), 'w') as f:
    f.write(class_report)