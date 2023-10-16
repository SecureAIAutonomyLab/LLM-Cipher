from torch import nn
import torch

# 3.9M Parameters
class SmallContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim1=1024, hidden_dim2=512, dropout=0.0):
        super(SmallContrastiveEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim2),
        )

    def forward(self, x):
        return self.network(x)


# ~10M Parameters
class MediumContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim1=2048, hidden_dim2=1536, hidden_dim3=1024, hidden_dim4=768, dropout=0.0):
        super(MediumContrastiveEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim4, 512),
        )

    def forward(self, x):
        return self.network(x)


# ~149M Parameters
# It's possible that a 'wider' architecture is necessary to really learn the differences in embeddings - brandon
# between machine and human
# we didn't really have time to properly test this
class LargeContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=2048, dropout=0.0):
        super(LargeContrastiveEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 8192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        return self.network(x)


class SimpleNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):

        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 2048)  
        self.fc3 = nn.Linear(2048, 1024)  
        self.fc4 = nn.Linear(1024, 512)   
        self.fc5 = nn.Linear(512, 256)   
        self.fc6 = nn.Linear(256, num_classes)  
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.fc6(x)
        return x


class SimpleNNRoBERTa(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):

        super(SimpleNNRoBERTa, self).__init__()
        
        self.fc1 = nn.Linear(768, 1536)
        self.fc2 = nn.Linear(1536, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.fc6(x)
        return x