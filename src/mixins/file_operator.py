import os
from dataclasses import dataclass
import pandas as pd
import torch
import pickle as pkl


@dataclass
class Paths:
    WORKDIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(WORKDIR, "data")


class FileOperatorMixin:

    def __init__(self):
        os.makedirs(Paths.DATA_PATH, exist_ok=True)

    def load(self, fname, dirname, use_cols=None, ext: str = "csv"):
        fpath = os.path.join(Paths.DATA_PATH, dirname, fname + f".{ext}")
        if ext == "csv":
            return pd.read_csv(fpath, usecols=use_cols, sep=";")
        elif ext == "pt":
            return torch.load(fpath)

    def save(self, data, fname, dirname, ext: str = "csv"):
        fpath = os.path.join(Paths.DATA_PATH, dirname, fname + f".{ext}")
        if ext == "csv":
            data.to_csv(fpath, index=False, sep=";")
        elif ext == "pt":
            torch.save(data, fpath)

    def get_all_dirs_in_data(self):
        return [d for d in os.listdir(Paths.DATA_PATH) if os.path.isdir(os.path.join(Paths.DATA_PATH, d))]

    def get_all_fnames(self, dirname: str, ext: str = "csv"):
        fpath = os.path.join(Paths.DATA_PATH, dirname)
        return [os.path.splitext(f)[0] for f in os.listdir(fpath) if f.endswith(ext)]

    def save_samples(self, samples: list[dict], fname: str):
        fpath = os.path.join(Paths.DATA_PATH, fname + ".pkl")
        with open(fpath, "wb") as f:
            pkl.dump(samples, f)

    def load_samples(self, fname: str) -> list[dict]:
        fpath = os.path.join(Paths.DATA_PATH, fname + ".pkl")
        with open(fpath, "rb") as f:
            return pkl.load(f)
