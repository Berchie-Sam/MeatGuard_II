import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from collections import Counter
from .transforms import build_train_transform, get_val_test_transform


def make_loaders(data_dir, strategy="standard", batch_size=32, seed=42):
    """
    Creates train, validation, and test loaders with strict benchmark parity.
    """
    # Seed for reproducibility
    torch.manual_seed(seed)
    
    # 1. Load dataset with training transform
    train_transform = build_train_transform(strategy)
    full_ds = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    n = len(full_ds)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    # Stratified split on raw indices
    full_raw_ds = datasets.ImageFolder(root=data_dir)
    indices = list(range(n))
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    def create_subset(indices, transform):
        class SubsetWithTransform(torch.utils.data.Dataset):
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform
            def __getitem__(self, idx):
                img, label = self.dataset.imgs[self.indices[idx]]
                img = Image.open(img).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, label
            def __len__(self):
                return len(self.indices)
        return SubsetWithTransform(full_raw_ds, indices, transform)

    clean_transform = get_val_test_transform()
    train_set = create_subset(train_idx, train_transform)
    val_set = create_subset(val_idx, clean_transform)
    test_set = create_subset(test_idx, clean_transform)
    
    # 4. Class Weights
    counts = Counter(full_raw_ds.targets)
    # weight = 1 / class_count, then normalized
    cw = torch.tensor([1.0/counts[i] for i in range(len(counts))], dtype=torch.float)
    cw = (cw / cw.sum())
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, cw
