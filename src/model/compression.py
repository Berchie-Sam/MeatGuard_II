import torch
import torch.nn as nn
import torch_pruning as tp


class MeatGuardPruner:
    """
    Structured channel pruning for MeatGuard-S that physically removes channels.

    Channels are ranked by the L1 norm of their weights and removed with Torch-Pruning's dependency
    graph, so coupled layers (residual additions, SE scaling, the classifier input) are pruned
    consistently and the result is a genuinely smaller dense network: fewer parameters, MACs and
    bytes, not just zeroed weights.

    The Stage-3 transformer block is excluded: CompactMHSA splits channels into 4 heads, and its
    256-d output is the knowledge-distillation alignment target. The classifier's output is kept.

    A pruned model no longer matches the MeatGuardS constructor, so save it with
    `torch.save(model, path)` (whole module) or `tp.state_dict(model)` / `tp.load_state_dict(model, ...)`.
    """
    def __init__(self, model, example_input=None):
        self.model = model
        self.example_input = example_input if example_input is not None else torch.randn(1, 3, 224, 224)

    def _ignored_layers(self):
        ignored = [self.model.classifier]
        if hasattr(self.model, "s3_attn"):
            ignored.append(self.model.s3_attn)
        return ignored

    def prune_conv_layers(self, amount=0.3):
        """Remove `amount` of the prunable channels in each layer group (L1-norm importance)."""
        device = next(self.model.parameters()).device
        example = self.example_input.to(device)
        before = self.stats(example)
        pruner = tp.pruner.MagnitudePruner(
            self.model,
            example_inputs=example,
            importance=tp.importance.MagnitudeImportance(p=1),
            pruning_ratio=amount,
            ignored_layers=self._ignored_layers(),
            round_to=8,                      # channel counts that mobile kernels handle efficiently
        )
        pruner.step()
        after = self.stats(example)
        print(f"Pruned {amount:.0%} of prunable channels: params {before['params_m']:.3f} M -> "
              f"{after['params_m']:.3f} M, MACs {before['gmacs']:.3f} G -> {after['gmacs']:.3f} G")
        return before, after

    def stats(self, example=None):
        example = self.example_input if example is None else example
        was_training = self.model.training
        self.model.eval()
        macs, params = tp.utils.count_ops_and_params(self.model, example.to(next(self.model.parameters()).device))
        self.model.train(was_training)
        return {"params_m": params / 1e6, "gmacs": macs / 1e9}

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
