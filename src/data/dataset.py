import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import build_train_transform, get_val_test_transform

SPLIT_FILE_NAME = "sessions.csv"


class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


def find_split_file(data_dir):
    """Look for sessions.csv in the data folder or its parent (E:/Datasets/full/{masked,sessions.csv})."""
    for candidate in (Path(data_dir) / SPLIT_FILE_NAME, Path(data_dir).parent / SPLIT_FILE_NAME):
        if candidate.exists():
            return candidate
    return None


def session_split(samples, data_dir, split_file):
    """Index lists from a session split CSV (columns: path, split), matched by class/stem."""
    table = pd.read_csv(split_file)
    key = lambda rel: Path(rel).with_suffix("").as_posix()
    split_of = dict(zip(table["path"].map(key), table["split"]))
    idx = {"train": [], "val": [], "test": []}
    missing = []
    for i, (path, _) in enumerate(samples):
        split = split_of.get(key(Path(path).relative_to(data_dir)))
        if split is None:
            missing.append(path)
        else:
            idx[split].append(i)
    if missing:
        raise ValueError(f"{len(missing)} images in {data_dir} are not in {split_file}, e.g. {missing[0]}")
    return idx["train"], idx["val"], idx["test"]


def random_split_indices(n, seed, ratios):
    warnings.warn(
        "No session split file found: falling back to a per-image random split. MeatScan contains "
        "near-identical burst frames, so this leaks scenes across train/val/test. "
        "Build one with `python -m src.data.sessions`.", stacklevel=3)
    indices = np.random.RandomState(seed).permutation(n).tolist()
    n_train, n_val = int(ratios[0] * n), int(ratios[1] * n)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def make_loaders(data_dir, strategy="standard", batch_size=32, seed=42,
                 split_file="auto", ratios=(0.70, 0.15, 0.15), num_workers=0, pin_memory=False):
    """
    Creates train, validation, and test loaders.

    split_file: path to a session split CSV (see src/data/sessions.py), "auto" to look for
    sessions.csv next to data_dir, or None to force the legacy per-image random split.
    Class weights are computed from the training split only.
    """
    torch.manual_seed(seed)
    data_dir = Path(data_dir)
    folder = datasets.ImageFolder(root=str(data_dir))
    samples = folder.samples

    if split_file == "auto":
        split_file = find_split_file(data_dir)
    if split_file:
        print(f"Using session split: {split_file}")
        train_idx, val_idx, test_idx = session_split(samples, data_dir, split_file)
    else:
        train_idx, val_idx, test_idx = random_split_indices(len(samples), seed, ratios)

    pick = lambda idx: [samples[i] for i in idx]
    clean_transform = get_val_test_transform()
    train_set = SubsetWithTransform(pick(train_idx), build_train_transform(strategy))
    val_set = SubsetWithTransform(pick(val_idx), clean_transform)
    test_set = SubsetWithTransform(pick(test_idx), clean_transform)

    # Class weights: inverse training-set frequency, normalised to sum to 1
    counts = Counter(label for _, label in train_set.samples)
    cw = torch.tensor([1.0 / counts[i] for i in range(len(folder.classes))], dtype=torch.float)
    cw = cw / cw.sum()

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
              persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **kw)
    val_loader = DataLoader(val_set, shuffle=False, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, cw
