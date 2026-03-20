from torch.utils.data import Dataset
import torch


class DatasetBuilder(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in self.data[idx].items()}
