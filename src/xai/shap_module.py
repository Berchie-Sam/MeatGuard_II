import torch
import numpy as np
from skimage.segmentation import slic
import shap

try:                                   # imported as src.xai.shap_module
    from ..data.transforms import MEAN, STD
except ImportError:                    # imported as xai.shap_module with src/ on sys.path (notebooks)
    from data.transforms import MEAN, STD


def _class_values(shap_values, class_idx):
    """SHAP values for one class as an array (1, C, H, W), across shap versions:
    older versions return a list with one array per class, newer ones an array with a trailing class axis."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx])
    vals = np.asarray(shap_values)
    return vals[..., class_idx] if vals.ndim == 5 else vals


class MeatGuardSHAP:
    """
    Implements SHAP (GradientExplainer, i.e. expected gradients) aggregated over SLIC superpixels
    for spatial attribution.
    """
    def __init__(self, model, background_images):
        self.model = model
        # Use a small set of background images to represent the baseline
        self.explainer = shap.GradientExplainer(self.model, background_images)

    def explain(self, input_tensor, n_samples=100):
        """
        Generates SHAP values for the input tensor.
        """
        shap_values = self.explainer.shap_values(input_tensor, nsamples=n_samples)
        return shap_values

    def superpixel_attribution(self, input_tensor, shap_values, labels=None, n_segments=50):
        """
        Aggregates SHAP values over SLIC superpixels to reduce noise.

        Returns (segments, superpixel values, per-pixel map): the map assigns every pixel the mean
        attribution of its superpixel, summed over colour channels.
        """
        # SLIC needs the real colours: undo the ImageNet normalisation before segmenting
        img = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        segments = slic(img_uint8, n_segments=n_segments, compactness=10, sigma=1, start_label=0)

        # Attribution per pixel: sum over colour channels of the class's SHAP values (1, C, H, W) -> (H, W)
        class_idx = 1 if labels is None else labels
        vals = _class_values(shap_values, class_idx)[0].sum(axis=0)

        unique_segs = np.unique(segments)
        superpixel_vals = np.array([vals[segments == s].mean() for s in unique_segs])
        pixel_map = np.zeros_like(vals)
        for s, v in zip(unique_segs, superpixel_vals):
            pixel_map[segments == s] = v
        return segments, superpixel_vals, pixel_map
