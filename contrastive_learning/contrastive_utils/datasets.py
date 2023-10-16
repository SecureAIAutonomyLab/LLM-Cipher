from torch.utils.data import Dataset
import numpy as np
import os
from itertools import combinations
import random
from itertools import permutations
from typing import List


class SymmetricContrastiveDataset(Dataset):
    def __init__(self, root_dir: str,
                 data_filter: str = '',
                 num_rows: int = 50):
        """
        Creates Dataset with pruned pair values for contrastive learning.
        Selects the 'inner square' of contrastive matrix.

        Args:
            root_dir (str): Path to data dir
            data_filter (str, optional): Filter npy dataset by word. 
                Example: "chatgpt" would select only the datasets associated with chatgpt.
                Defaults to ''.
            num_rows (int, optional): The number of rows to slice from all selected npy files. Defaults to 50.
        """
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.data_filter = data_filter

        human_files = sorted([file for file in os.listdir(
            self.root_dir) if 'human' in file and self.data_filter in file])
        machine_files = sorted([file for file in os.listdir(
            self.root_dir) if 'machine' in file and self.data_filter in file])

        # Loading data and labels
        for human_file, machine_file in zip(human_files, machine_files):
            human_embeddings = np.load(
                os.path.join(root_dir, human_file))[:num_rows]
            machine_embeddings = np.load(
                os.path.join(root_dir, machine_file))[:num_rows]

            self.data.append(human_embeddings)
            self.data.append(machine_embeddings)
            self.labels.extend([0] * len(human_embeddings))  # 0 for human
            self.labels.extend([1] * len(machine_embeddings))  # 1 for machine

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)
        self.pairs = self.generate_pairs()

    def generate_pairs(self):
        pairs = []
        human_indices = [i for i, label in enumerate(
            self.labels) if label == 0]
        machine_indices = [i for i, label in enumerate(
            self.labels) if label == 1]

        # Expanding possible positive pairs
        pairs.extend(combinations(human_indices, 2))
        pairs.extend(combinations(machine_indices, 2))

        # Negative pairs, human with machine
        for hi in human_indices:
            for mi in machine_indices:
                pairs.append((hi, mi))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        xi, xj = self.data[i], self.data[j]
        label_i, label_j = self.labels[i], self.labels[j]
        return (xi, xj, label_i, label_j)


class PostivePairDataset(Dataset):
    def __init__(self, root_dir: str, data_filter: str = '', num_rows: int = 50):
        """
        Args:
            root_dir (str): Path to data dir
            data_filter (str, optional): Filter npy dataset by word.
            num_rows (int, optional): The number of rows to slice from all selected npy files.
        """
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        human_files = sorted([file for file in os.listdir(
            self.root_dir) if 'human' in file and data_filter in file])
        machine_files = sorted([file for file in os.listdir(
            self.root_dir) if 'machine' in file and data_filter in file])

        for hfile, mfile, in zip(human_files, machine_files):
            print(hfile)
            print(mfile)

        # Loading data and labels
        for human_file, machine_file in zip(human_files, machine_files):
            human_embeddings = np.load(
                os.path.join(root_dir, human_file))[:num_rows]
            machine_embeddings = np.load(
                os.path.join(root_dir, machine_file))[:num_rows]

            self.data.append(human_embeddings)
            self.data.append(machine_embeddings)
            self.labels.extend([0] * len(human_embeddings))
            self.labels.extend([1] * len(machine_embeddings))

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        # Assuming each embedding will be paired with another random one
        return len(self.data)

    def __getitem__(self, idx):
        xi, label_i = self.data[idx], self.labels[idx]

        # Get positive pair
        if label_i == 0:  # if human
            pos_indices = [i for i, label in enumerate(
                self.labels) if label == 0 and i != idx]
        else:  # if machine
            pos_indices = [i for i, label in enumerate(
                self.labels) if label == 1 and i != idx]

        paired_idx = random.choice(pos_indices)
        xj, label_j = self.data[paired_idx], self.labels[paired_idx]

        return (xi, xj, label_i, label_j)


class TripletDataset(Dataset):
    def __init__(self, root_dir: str,
                 data_filter: str = '',
                 num_rows: int = 50,
                 max_samples: int = 10000):

        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.data_filter = data_filter
        self.max_samples = max_samples

        human_files = sorted([file for file in os.listdir(
            self.root_dir) if 'human' in file and self.data_filter in file])
        machine_files = sorted([file for file in os.listdir(
            self.root_dir) if 'machine' in file and self.data_filter in file])

        # Loading data and labels
        for human_file, machine_file in zip(human_files, machine_files):
            human_embeddings = np.load(
                os.path.join(root_dir, human_file))[:num_rows]
            machine_embeddings = np.load(
                os.path.join(root_dir, machine_file))[:num_rows]
            self.data.append(human_embeddings)
            self.data.append(machine_embeddings)
            self.labels.extend([0] * len(human_embeddings))  # 0 for human
            self.labels.extend([1] * len(machine_embeddings))  # 1 for machine

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)
        self.triplets = self.generate_triplets()
        self.datasets_used = human_files + machine_files

    def generate_triplets(self):
        triplets = set()  # Using set to avoid duplicates
        human_indices = [i for i, label in enumerate(
            self.labels) if label == 0]
        machine_indices = [i for i, label in enumerate(
            self.labels) if label == 1]

        while len(triplets) < self.max_samples:
            if random.random() < 0.5:  # Select human-human-machine
                anchor_idx, positive_idx = random.sample(human_indices, 2)
                negative_idx = random.choice(machine_indices)
            else:  # Select machine-machine-human
                anchor_idx, positive_idx = random.sample(machine_indices, 2)
                negative_idx = random.choice(human_indices)

            triplet = (anchor_idx, positive_idx, negative_idx)
            triplets.add(triplet)

        return list(triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        anchor, positive, negative = self.data[anchor_idx], self.data[positive_idx], self.data[negative_idx]
        return (anchor, positive, negative)


class TripletDatasetSetAnchored(Dataset):
    def __init__(self, root_dir: str, data_filter: str = '', train_ratio=0.8, heldout=False, **kwargs):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.data_filter = data_filter

        if heldout:
            human_files = sorted([file for file in os.listdir(
                self.root_dir) if 'human' in file and self.data_filter not in file])
            machine_files = sorted([file for file in os.listdir(
                self.root_dir) if 'machine' in file and self.data_filter not in file])
        else:
            human_files = sorted([file for file in os.listdir(
                self.root_dir) if 'human' in file and self.data_filter in file])
            machine_files = sorted([file for file in os.listdir(
                self.root_dir) if 'machine' in file and self.data_filter in file])

        for human_file, machine_file in zip(human_files, machine_files):
            human_embeddings = np.load(
                os.path.join(root_dir, human_file))
            machine_embeddings = np.load(
                os.path.join(root_dir, machine_file))

            human_slice_amount = int(train_ratio * len(human_embeddings))
            machine_slice_amount = int(train_ratio * len(machine_embeddings))

            human_embeddings = human_embeddings[:human_slice_amount]
            machine_embeddings = machine_embeddings[:machine_slice_amount]

            self.data.append(human_embeddings)
            self.data.append(machine_embeddings)

            self.labels.extend([0] * len(human_embeddings))  # 0 for human
            self.labels.extend([1] * len(machine_embeddings))  # 1 for machine

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)
        self.datasets_used = human_files + machine_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        while True:
            positive_idx = random.randint(0, len(self.data) - 1)
            if self.labels[positive_idx] == anchor_label and positive_idx != idx:
                break

        while True:
            negative_idx = random.randint(0, len(self.data) - 1)
            if self.labels[negative_idx] != anchor_label:
                break

        positive = self.data[positive_idx]
        negative = self.data[negative_idx]

        return (anchor, positive, negative)


class ClassificationDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 data_filter: str = '', train_ratio=0.9, heldout=False, two_class=False, **kwargs):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.train_ratio = train_ratio
        self.exclude_filters = ['llama', 'flant5']
        self.included_models = ['chatgpt',
                                'bloomz', 'davinci', 'dolly', 'cohere']

        if heldout and data_filter in self.included_models:
            self.included_models.remove(data_filter)

        if data_filter:
            self.exclude_filters.append(data_filter)

        self.label_mapping = self.generate_label_mapping(two_class)

        self.load_and_process_files()

    def generate_label_mapping(self, two_class):
        if two_class:
            return {model: 1 for model in self.included_models}
        else:
            return {model: idx for idx, model in enumerate(sorted(self.included_models), start=1)}

    def file_filter(self, filename, is_human=True):
        if any(excl in filename for excl in self.exclude_filters):
            return False

        if is_human:
            return 'human' in filename

        # machine files
        for model in self.included_models:
            if model in filename:
                return 'machine' in filename
        return False

    def load_and_process_files(self):
        human_files = sorted([file for file in os.listdir(
            self.root_dir) if self.file_filter(file, is_human=True)])
        machine_files = sorted([file for file in os.listdir(
            self.root_dir) if self.file_filter(file, is_human=False)])

        for human_file in human_files:
            embeddings = np.load(os.path.join(self.root_dir, human_file))
            slice_amount = int(self.train_ratio * len(embeddings))
            embeddings = embeddings[:slice_amount]

            self.data.append(embeddings)
            self.labels.extend([0] * len(embeddings))

        for machine_file in machine_files:
            embeddings = np.load(os.path.join(self.root_dir, machine_file))
            slice_amount = int(self.train_ratio * len(embeddings))
            embeddings = embeddings[:slice_amount]

            label = self.get_label_for_file(machine_file)
            if label is not None:
                self.data.append(embeddings)
                self.labels.extend([label] * len(embeddings))

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)
        self.datasets_used = human_files + machine_files

    def get_label_for_file(self, filename):
        for key, value in self.label_mapping.items():
            if key in filename:
                return value
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class InverseClassificationDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 # Expect a list of keywords to include
                 include_filters: List[str],
                 include_ratio=0.1,
                 two_class=False, **kwargs):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.include_ratio = include_ratio
        self.include_filters = include_filters
        self.included_models = ['chatgpt',
                                'bloomz', 'davinci', 'dolly', 'cohere']

        self.label_mapping = self.generate_label_mapping(two_class)

        self.load_and_process_files()

    def generate_label_mapping(self, two_class):
        if two_class:
            return {model: 1 for model in self.included_models}
        else:
            return {model: idx for idx, model in enumerate(sorted(self.included_models), start=1)}

    def file_filter(self, filename):
        return any(incl in filename for incl in self.include_filters)

    def load_and_process_files(self):
        all_files = sorted([file for file in os.listdir(
            self.root_dir) if self.file_filter(file)])

        for file in all_files:
            embeddings = np.load(os.path.join(self.root_dir, file))
            slice_amount = int(self.include_ratio * len(embeddings))
            # Take the last portion of the data
            embeddings = embeddings[-slice_amount:]

            label = 0 if 'human' in file else self.get_label_for_file(file)
            if label is not None:
                self.data.append(embeddings)
                self.labels.extend([label] * len(embeddings))

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)
        self.datasets_used = all_files

    def get_label_for_file(self, filename):
        for key, value in self.label_mapping.items():
            if key in filename:
                return value
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class InTheWild(Dataset):
    def __init__(self, root_dir: str, keyword=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.keyword = keyword
        self.load_and_process_files()

    def load_and_process_files(self):
        all_files = [file for file in os.listdir(self.root_dir) if (
            'human' in file or 'machine' in file) and self.keyword in file]

        for file in all_files:
            embeddings = np.load(os.path.join(self.root_dir, file))

            label = 0 if 'human' in file else 1

            self.data.append(embeddings)
            self.labels.extend([label] * len(embeddings))

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Adversarial(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        self.load_and_process_files()

    def load_and_process_files(self):
        machine_files = [file for file in os.listdir(
            self.root_dir) if 'machine' in file]

        for file in machine_files:
            embeddings = np.load(os.path.join(self.root_dir, file))
            self.data.append(embeddings)
            self.labels.extend([1] * len(embeddings))

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
