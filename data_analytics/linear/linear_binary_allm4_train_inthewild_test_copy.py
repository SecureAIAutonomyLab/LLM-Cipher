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
results_folder = "../../results/linear_results"
classes = ['human', 'machine']

# Load and preprocess data
def load_and_preprocess_data(folder_name, classes):
    embeddings_list = []
    labels_list = []
    file_dict = {class_name: [] for class_name in classes}  # Dictionary to store file names for each class
    
    for class_idx, class_name in enumerate(classes):
        for file_name in os.listdir(folder_name):
            # Condition to check if file should be loaded into the current class
            if class_name in file_name:
                load_condition = True
            else:
                load_condition = False
            
            # Load file if condition is met
            if load_condition:
                file_path = os.path.join(folder_name, file_name)
                embeddings = np.load(file_path)
                
                # Add file name to the dictionary
                file_dict[class_name].append(file_name)
                
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
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 2048)  
        self.fc3 = nn.Linear(2048, 1024)  
        self.fc4 = nn.Linear(1024, 512)   
        self.fc5 = nn.Linear(512, 256)   
        self.fc6 = nn.Linear(256, 2)      
        
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
best_model_path = os.path.join(results_folder, 'best_model_linear_binary_2.pth')  # Path to save the best model

train_losses = []  # List to store training loss at each epoch
val_losses = []  # List to store validation loss at each epoch

train_losses = []  # List to store training loss at each epoch
val_losses = []  # List to store validation loss at each epoch

# Wrap your epochs with tqdm for a progress bar
for epoch in tqdm(range(500), desc="Training", unit="epoch"):
    model.train()  # Set the model to training mode
    
    # Initialize variables to store loss during this epoch
    train_loss_total = 0.0
    train_samples = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        # Accumulate the loss and number of samples
        train_loss_total += loss.item() * X_batch.size(0)
        train_samples += X_batch.size(0)
    
    # Compute average training loss for the epoch
    train_loss_avg = train_loss_total / train_samples
    train_losses.append(train_loss_avg)  # Append average training loss to the list
    
    # Validate the model
    model.eval()  # Set the model to evaluation mode
    val_loss_total = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
            val_outputs = model(X_batch)
            val_loss = criterion(val_outputs, y_batch)
            val_loss_total += val_loss.item() * X_batch.size(0)
            total += y_batch.size(0)
            
            _, predicted = torch.max(val_outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
    
    val_accuracy = correct / total
    val_losses.append(val_loss_total / total)  # Compute average validation loss
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1}, Validation Accuracy Improved: {best_val_accuracy:.4f}, Model Saved!")

# Plot training and validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(results_folder, 'train_val_loss_plot_linear_binary_2.png'))  # Save plot to file
plt.show()  # Display plot            

# Load the best model for evaluation and further testing
model.load_state_dict(torch.load(best_model_path))


# Evaluate the model
X_test, y_test = X_test.to(device), y_test.to(device)
test_outputs = model(X_test)
_, predicted = torch.max(test_outputs, 1)

avg_logits = {}
std_logits = {}

for class_idx, class_name in enumerate(classes):
    class_logits = test_outputs[predicted == class_idx, class_idx].cpu().detach().numpy()  # Extract logits of predicted class_idx
    if len(class_logits) != 0:
        avg_logits[class_name] = np.mean(class_logits)
        std_logits[class_name] = np.std(class_logits)
    else:
        avg_logits[class_name] = np.nan  # or use some placeholder or ignore
        std_logits[class_name] = np.nan  # or use some placeholder or ignore

# Move tensors to CPU and convert to NumPy before passing to confusion_matrix
y_test_np = y_test.cpu().numpy()
predicted_np = predicted.cpu().numpy()

conf_matrix = confusion_matrix(y_test_np, predicted_np)
class_report = classification_report(y_test_np, predicted_np, target_names=classes, zero_division=1)

# Save the confusion matrix and classification report
with open(os.path.join(results_folder, 'classification_report_linear_binary_2.txt'), 'w') as f:
    f.write(str(conf_matrix))
    f.write('\n')
    f.write(class_report)
    f.write('\n\n')
    for class_name in classes:
        f.write(f'Average Logit for class {class_name}: {avg_logits[class_name]:.4f}\n')
        f.write(f'Standard Deviation of Logit for class {class_name}: {std_logits[class_name]:.4f}\n')
    f.write('\n')

    

# Set up the in-the-wild test folder
in_the_wild_folder = "../../data/embedding_data_T5_inthewild_npy"

# Load and preprocess in-the-wild data
def load_in_the_wild_data(folder_name, classes):
    dataset_stats = {}
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        embeddings = np.load(file_path)
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
with open(os.path.join(results_folder, 'in_the_wild_stats_linear_binary_2.txt'), 'w') as f:
    for dataset, stats in wild_stats.items():
        f.write(f"Statistics for {dataset}:\n")
        for class_name in classes:
            count = stats['counts'].get(class_name, 0)
            avg_logit = stats['avg_logits'].get(class_name, float('nan'))
            std_logit = stats['std_logits'].get(class_name, float('nan'))
            f.write(f"{class_name}: {count}, Average Logit: {avg_logit}, Std Dev Logit: {std_logit}\n")
        f.write("-------------------------------\n")
