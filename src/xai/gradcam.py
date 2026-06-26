import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

class GradCAM:
    """
    Implements Grad-CAM for spatial localization of meat spoilage features.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = self._register_hooks()

    def _register_hooks(self):
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activations(module, input, output):
            self.activations = output

        # Register hooks on the target layer
        target = dict(self.model.named_modules()).get(self.target_layer)
        if target is None:
            raise ValueError(f"Layer {self.target_layer} not found in model.")
            
        handle_act = target.register_forward_hook(save_activations)
        handle_grad = target.register_full_backward_hook(save_gradients)
        return (handle_act, handle_grad)

    def generate(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap for a given input.
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0][class_idx] = 1
        logits.backward(gradient=one_hot)

        # Global average pool of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        # ReLU on the map to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()

    def apply_overlay(self, img_tensor, cam_map, alpha=0.5):
        """
        Overlays the heatmap on the original image.
        """
        # Convert tensor to numpy (H, W, C)
        img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # Unnormalize (approximation)
        img = (img * 0.229) + 0.485 # simplified for vis
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        # Resize heatmap to match image
        heatmap = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
