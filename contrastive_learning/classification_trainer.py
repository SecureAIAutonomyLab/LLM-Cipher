import torch
import torch.optim as optim
from tqdm import tqdm
from contrastive_utils.datasets import TripletDataset, TripletDatasetSetAnchored, ClassificationDataset
from contrastive_utils.models import SimpleNN 
from contrastive_utils.loss_functions import CosineContrastiveLoss, EuclideanContrastiveLoss, NTXentLoss
from torch.utils.data.dataloader import DataLoader
import os
import matplotlib.pyplot as plt
import json
from torch.nn import TripletMarginLoss
from torch.nn import CrossEntropyLoss

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.encoder = config["model"].to(self.device)
        self.criterion = config["criterion"]
        self.optimizer = config["optimizer"]
        # self.scheduler = config["scheduler"]
        self.train_loader = config["train_loader"]
        self.val_loader = config["val_loader"]
        self.epochs = config["epochs"]
        self.save_dir = config["save_dir"]
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.steps = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def gather_metadata(self):
        return {
            "config": {
                "epochs": self.epochs,
                "batch_size": self.config.get("batch_size", "N/A"),
                "model": str(self.encoder),
                "criterion": str(self.criterion),
                "optimizer": str(self.optimizer),
                "end_margin": self.criterion.margin if isinstance(self.criterion, TripletMarginLoss) else None,
                "steps_per_batch_train": len(self.train_loader),
                # "scheduler": str(self.scheduler),
            },
            "dataset_sizes": {
                "total": len(self.train_loader.dataset) + len(self.val_loader.dataset),
                "train": len(self.train_loader.dataset),
                "validation": len(self.val_loader.dataset)
            }
        }

    def train_epoch(self):
        total_train_loss = 0.0
        self.encoder.train()

        pbar = tqdm(self.train_loader, desc="Training", dynamic_ncols=True)

        for (inputs, labels) in pbar:
            self.optimizer.zero_grad()

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.encoder(inputs)

            loss = self.criterion(outputs, labels)
            total_train_loss += loss.item()

            pbar.set_postfix({"Training Loss": loss.item()})

            loss.backward()
            self.optimizer.step()

            self.steps += 1
            if self.steps % 100 == 0:
                self.train_loss_history.append(loss.item())

        return total_train_loss / len(self.train_loader)

    def validate_epoch(self):
        total_val_loss = 0.0
        self.encoder.eval()

        pbar = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True)

        with torch.no_grad():
            for (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.encoder(inputs)

                val_loss = self.criterion(outputs, labels)
                total_val_loss += val_loss.item()

                pbar.set_postfix({"Validation Loss": val_loss.item()})

        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss

    def save_model(self, avg_val_loss):
        if avg_val_loss <= self.best_val_loss:
            self.best_val_loss = avg_val_loss
            model_path = os.path.join(
                self.save_dir, 'contrastive_model_best.pt')
            torch.save(self.encoder.state_dict(), model_path)

    def train(self):
        print('Starting training!')
        for epoch in range(self.epochs):
            self.encoder.train()
            avg_train_loss = self.train_epoch()
            print(
                f'\nTraining Loss after Epoch {epoch+1}: {avg_train_loss:.4f}\n')

            self.encoder.eval()
            avg_val_loss = self.validate_epoch()
            self.val_loss_history.append(avg_val_loss)
            print(
                f'\nValidation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}\n')

            self.save_model(avg_val_loss)

        self.save_training_data()

    def save_training_data(self):
        metadata = self.gather_metadata()

        data_to_save = {
            "metadata": metadata,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history
        }
        with open(os.path.join(self.save_dir, 'training_data.json'), 'w') as f:
            json.dump(data_to_save, f, indent=4)

        steps_per_epoch = len(self.train_loader)

        val_loss_steps = [
            (epoch+1) * steps_per_epoch for epoch in range(len(self.val_loss_history))]
        train_loss_steps = [
            i * 100 for i in range(1, len(self.train_loss_history) + 1)]

        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_steps, self.train_loss_history,
                 label='Training Loss')
        plt.plot(val_loss_steps, self.val_loss_history,
                 label='Validation Loss')
        plt.xlabel('Batch Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'loss_history_plot.png'))
        plt.close()


models = {
    # "Small": SmallContrastiveEncoder,
    # "Medium": MediumContrastiveEncoder,
    # "Large": LargeContrastiveEncoder
    'ClassificationDNN' : SimpleNN
}

data_filters = [
    'allm4_2class',
    'allm4_6class',
    'chatgpt_heldout_5class',
    'bloomz_heldout_5class',
    'cohere_heldout_5class',
    'davinci_heldout_5class',
    'dolly_heldout_5class',
    'peerread_heldout_6class',
    'reddit_heldout_6class',
    'wikihow_heldout_6class',
    'wikipedia_heldout_6class',
    'arxiv_heldout_6class',
    'chatgpt_heldout_2class',
    'bloomz_heldout_2class',
    'cohere_heldout_2class',
    'davinci_heldout_2class',
    'dolly_heldout_2class',
    'peerread_heldout_2class',
    'reddit_heldout_2class',
    'wikihow_heldout_2class',
    'wikipedia_heldout_2class',
    'arxiv_heldout_2class',
]

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for data_filter in data_filters:

        num_classes = 6 
        two_class = False
        folder_name = data_filter

        if data_filter == 'allm4_6class':
            num_classes = 6

        elif data_filter == 'allm4_2class':
            num_classes = 2
            two_class = True

        elif '2class' in data_filter:
            num_classes = 2
            two_class = True

        elif '5class' in data_filter:
            num_classes = 5

        heldout = False
        if 'heldout' in data_filter:
            heldout = True
            data_filter = data_filter.split('_')[0]

        symmetric_dataset = ClassificationDataset(
            '../data/embedding_data_T5_npy/', 
            data_filter=data_filter,
            heldout=heldout,
            two_class=two_class,
        )

        train_size = int(0.89 * len(symmetric_dataset))
        val_size = len(symmetric_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            symmetric_dataset, [train_size, val_size])

        print(f'Total Dataset Size: {len(symmetric_dataset)}')
        print(f'Training Dataset Size: {train_size}')
        print(f'Validation Dataset Size: {val_size}')
        print(f'Using Dataset Filter Word: "{data_filter}"')
        print(f'Using the following dataset files:')

        for file_name in symmetric_dataset.datasets_used:
            print(file_name)

        save_name = f'{folder_name}'

        for model_name, model_class in models.items():
            config = {
                "device": device,
                "epochs": 500,
                "batch_size": 512,
                "model": model_class(num_classes=num_classes).to(device),
                "criterion": CrossEntropyLoss(),
                "save_dir": f"results_0/{save_name}/",
                "dataset_files_used": symmetric_dataset.datasets_used
            }

            config['optimizer'] = torch.optim.Adam(config['model'].parameters(), lr=0.0001) 
            config['train_loader'] = DataLoader(
                train_dataset, batch_size=config['batch_size'], shuffle=True)
            config['val_loader'] = DataLoader(
                val_dataset, batch_size=config['batch_size'], shuffle=True)

            trainer = Trainer(config)
            trainer.train()

            del trainer
            del config
