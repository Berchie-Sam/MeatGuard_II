import torch
import numpy as np
from skimage.segmentation import slic
import shap

class MeatGuardSHAP:
    """
    Implements SHAP (GradientExplainer) using SLIC superpixels for
    channel-wise and spatial attribution.
    """
    def __init__(self, model, background_images):
        self.model = model
        # Use a small set of background images to represent the baseline
        self.explainer = shap.GradientExplainer(self.model, background_images)

    def explain(self, input_tensor, n_samples=100):
        """
        Generates SHAP values for the input tensor.
        """
        shap_values = self.explainer.shap_values(input_tensor, n_samples=n_samples)
        return shap_values

    def superpixel_attribution(self, input_tensor, shap_values, labels=None):
        """
        Aggregates SHAP values over SLIC superpixels to reduce noise.
        """
        img = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        
        # SLIC Segmentation
        segments = slic(img_uint8, n_segments=50, compactness=10, sigma=1)
        
        # Map SHAP values to superpixels
        # shap_values is typically a list [class0_vals, class1_vals]
        # We take the values for the predicted class (usually index 1 for spoiled)
        class_idx = 1 if labels is None else labels
        vals = shap_values[class_idx].reshape(img.shape)
        
        unique_segs = np.unique(segments)
        superpixel_vals = np.zeros(len(unique_segs))
        for i, seg_id in enumerate(unique_segs):
            mask = (segments == seg_id)
            superpixel_vals[i] = np.mean(vals[mask])
            
        return segments, superpixel_vals

