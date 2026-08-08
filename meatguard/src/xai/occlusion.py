import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class OcclusionSensitivity:
    """
    Implements Occlusion Sensitivity by sliding a mask over the image.
    """
    def __init__(self, model, patch_size=16, stride=8):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride

    def generate_map(self, input_tensor, class_idx=None):
        self.model.eval()
        B, C, H, W = input_tensor.shape
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = baseline_logits.argmax(dim=1).item()
            baseline_conf = F.softmax(baseline_logits, dim=1)[0, class_idx].item()

        # Initialize occlusion map
        occ_map = np.zeros((H, W))

        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                # Create occluded image
                occluded_img = input_tensor.clone()
                occluded_img[:, :, y:y+self.patch_size, x:x+self.patch_size] = 0 # Black mask
                
                with torch.no_grad():
                    logits = self.model(occluded_img)
                    conf = F.softmax(logits, dim=1)[0, class_idx].item()
                    
                # Drop in confidence = importance
                occ_map[y:y+self.patch_size, x:x+self.patch_size] += (baseline_conf - conf)

        # Normalize map to [0, 1]
        occ_map -= occ_map.min()
        occ_map /= (occ_map.max() + 1e-7)
        
        return occ_map
