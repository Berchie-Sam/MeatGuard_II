import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_insertion_auc(model, image, mask, target_class):
    """
    Measures model confidence as pixels from the mask are added incrementally.
    """
    B, C, H, W = image.shape
    # Flatten mask and get indices sorted by importance
    mask_flat = mask.flatten()
    indices = np.argsort(mask_flat)[::-1]
    
    confidences = []
    # Incremental additive process
    current_img = torch.zeros_like(image)
    
    # Step size (e.g., 10% of pixels per step)
    step = int(len(indices) * 0.1)
    
    for i in range(0, len(indices), step):
        # Add top pixels
        idx_subset = indices[i : i + step]
        # Map flat index back to 2D
        for idx in idx_subset:
            y, x = divmod(idx, W)
            current_img[:, :, y, x] = image[:, :, y, x]
        
        with torch.no_grad():
            logits = model(current_img)
            conf = torch.softmax(logits, dim=1)[0, target_class].item()
            confidences.append(conf)
            
    # Calculate AUC of confidence curve
    # Note: This is a simplification. True Insertion AUC uses trapezoidal integration.
    return np.trapz(confidences, dx=1.0)

def calculate_deletion_auc(model, image, mask, target_class):
    """
    Measures confidence drop as top-importance pixels are removed.
    """
    B, C, H, W = image.shape
    mask_flat = mask.flatten()
    indices = np.argsort(mask_flat)[::-1]
    
    confidences = []
    current_img = image.clone()
    
    step = int(len(indices) * 0.1)
    for i in range(0, len(indices), step):
        idx_subset = indices[i : i + step]
        for idx in idx_subset:
            y, x = divmod(idx, W)
            current_img[:, :, y, x] = 0 # Occlude
            
        with torch.no_grad():
            logits = model(current_img)
            conf = torch.softmax(logits, dim=1)[0, target_class].item()
            confidences.append(conf)
            
    return np.trapz(confidences, dx=1.0)

def calculate_iou(mask_a, mask_b):
    """
    Intersection over Union between two binary masks.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / (union + 1e-7)
