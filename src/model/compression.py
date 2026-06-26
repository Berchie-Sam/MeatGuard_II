import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class MeatGuardPruner:
    """
    Implements L1-norm structured channel pruning for MeatGuard-S.
    Target: Remove ~30% of filters in convolutional layers.
    """
    def __init__(self, model):
        self.model = model

    def prune_conv_layers(self, amount=0.3):
        """
        Prunes channels in Conv2d layers using L1 norm.
        """
        for name, module in self.model.named_modules():
            # Prune only Conv2d layers that are not 1x1 (preserving projections)
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
                prune.ln_structured(
                    module, 
                    name="weight", 
                    amount=amount, 
                    n=2, # L1 norm
                    dim=0 # Prune channels
                )
                # Make pruning permanent by removing the mask
                prune.remove(module, 'weight')
        
        print(f"Successfully applied L1-structured pruning ({amount*100}% channels) to Conv layers.")

    def recover_fine_tune(self, train_loader, val_loader, criterion, optimizer, device="cuda", epochs=10):
        """
        Performs a short fine-tuning run to recover accuracy after pruning.
        """
        self.model.to(device)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Recovery Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
