from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random

random.seed(101)


class WheelDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _to_scalar_speed(speed) -> torch.Tensor:
        if torch.is_tensor(speed):
            speed_tensor = speed.float()
        else:
            speed_tensor = torch.as_tensor(speed, dtype=torch.float32)

        if speed_tensor.ndim > 0:
            speed_tensor = speed_tensor.mean()

        return speed_tensor

    def __getitem__(self, idx):
        sample = self.data[idx]

        x = torch.tensor(sample["X"], dtype=torch.float32)
        speed = self._to_scalar_speed(sample["speed"])

        target = torch.tensor(sample["target"], dtype=torch.long)

        return {
            "X": x,
            "speed": speed,
            "target": target,
        }


class DataProcessor:

    @staticmethod
    def collate_fn(batch):

        signals = [item["X"] for item in batch]
        speeds = torch.stack([item["speed"] for item in batch])  # (B,)
        targets = torch.stack([item["target"] for item in batch])  # (B,)

        lengths = torch.tensor([len(x) for x in signals], dtype=torch.long)

        padded_signals = pad_sequence(signals, batch_first=True, padding_value=0.0)

        return {
            "signal": padded_signals,
            "speed": speeds,
            "target": targets,
            "lengths": lengths,
        }

    def train_test_split(self, data: list[dict], train_size: float = 0.8, shuffle: bool = True):
        data_len = len(data)
        indices = list(range(data_len))
        if shuffle:
            random.shuffle(indices)
        split_idx = int(train_size * data_len)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        return train_data, test_data

    def build_datasets(self, data: list[dict], train_size: float = 0.8):
        train_data, test_data = self.train_test_split(data, train_size)
        return WheelDataset(train_data), WheelDataset(test_data)

    def build_dataloaders(self, train_dataset: Dataset, test_dataset: Dataset, batch_size: int = 32):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        return train_loader, test_loader

    def process(self, data: list[dict], train_size: float = 0.8, batch_size: int = 32):
        train_dataset, test_dataset = self.build_datasets(data, train_size)
        return self.build_dataloaders(train_dataset, test_dataset, batch_size)
