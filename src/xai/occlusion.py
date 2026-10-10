import torch
import torch.nn.functional as F
import numpy as np


class OcclusionSensitivity:
    """
    Implements Occlusion Sensitivity by sliding a mask over the image.

    Each patch is set to 0 in normalised space (the ImageNet mean colour). A pixel's score is the drop in
    the class probability averaged over all patch positions that cover it, so pixels covered by more
    patches are not favoured.
    """
    def __init__(self, model, patch_size=16, stride=8, batch_size=64):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size

    @torch.no_grad()
    def generate_map(self, input_tensor, class_idx=None, normalize=True):
        """normalize=False returns the raw mean probability drop (positive = supports the class)."""
        self.model.eval()
        B, C, H, W = input_tensor.shape
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        baseline_conf = F.softmax(logits, dim=1)[0, class_idx].item()

        positions = [(y, x) for y in range(0, H - self.patch_size + 1, self.stride)
                     for x in range(0, W - self.patch_size + 1, self.stride)]
        drop_sum = np.zeros((H, W))
        count = np.zeros((H, W))
        for i in range(0, len(positions), self.batch_size):
            chunk = positions[i:i + self.batch_size]
            batch = input_tensor.repeat(len(chunk), 1, 1, 1)
            for k, (y, x) in enumerate(chunk):
                batch[k, :, y:y + self.patch_size, x:x + self.patch_size] = 0
            confs = F.softmax(self.model(batch), dim=1)[:, class_idx].cpu().numpy()
            for (y, x), conf in zip(chunk, confs):
                drop_sum[y:y + self.patch_size, x:x + self.patch_size] += baseline_conf - conf
                count[y:y + self.patch_size, x:x + self.patch_size] += 1
        occ_map = np.divide(drop_sum, count, out=np.zeros_like(drop_sum), where=count > 0)
        if not normalize:
            return occ_map

        # Normalize map to [0, 1] (for display)
        occ_map -= occ_map.min()
        occ_map /= (occ_map.max() + 1e-7)
        return occ_map
