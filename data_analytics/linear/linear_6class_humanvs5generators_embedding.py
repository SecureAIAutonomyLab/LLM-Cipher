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
train_folder_name = "../../data/embedding_data_T5_npy"
results_folder = "../../results/linear_results/embeddings"
model_weights = "../../results/linear_results/best_model_humanvs5generators.pth"
classes = ['bloomz', 'chatgpt', 'cohere', 'davinci', 'dolly', 'human']

# Load and preprocess data
def load_and_preprocess_data(folder_name, classes):
    embeddings_list = []
    labels_list = []
    file_dict = {class_name: [] for class_name in classes}  # Dictionary to store file names for each class
    
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
                
                # Print the number of samples in the current .npy file
                #print(f"Number of samples in {file_name}: {embeddings.shape[0]}")
                
                # Add file name to the dictionary
                file_dict[class_name].append(file_name)
                
                labels = np.full(embeddings.shape[0], class_idx)
                embeddings_list.append(embeddings)
                labels_list.append(labels)
    
    # Print file names associated with each class
#     for class_name, files in file_dict.items():
#         print(f"Files for class {class_name}: {', '.join(files)}")
    
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

# Print number of samples in each class for train, val, and test splits
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_val, counts_val = np.unique(y_val, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)


# Convert numpy arrays to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)


# Create DataLoader for train, validation, and test sets
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define Neural Network
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

    def extract_embedding(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        embedding = torch.relu(self.fc5(x))
        return embedding

model = SimpleNN().to(device)
model.load_state_dict(torch.load(model_weights))

def save_embeddings(folder_name, results_folder, classes, model):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Ensure no gradients are computed
        for class_name in classes:
            for file_name in os.listdir(folder_name):
                if (class_name == 'human' and class_name in file_name) or \
                   (class_name != 'human' and class_name in file_name and 'machine' in file_name):
                    file_path = os.path.join(folder_name, file_name)
                    data = np.load(file_path)
                    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                    embeddings = model.extract_embedding(data_tensor)
                    embeddings_np = embeddings.cpu().numpy()
                    save_path = os.path.join(results_folder, file_name)
                    np.save(save_path, embeddings_np)

save_embeddings(train_folder_name, results_folder, classes, model)

