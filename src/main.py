import argparse
import torch
from utils.config_manager import ConfigManager
from model.architecture import MeatGuardS
from model.trainer import MeatGuardTrainer
from data.dataset import make_loaders
from utils.viz_manager import VisualizationManager
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--mode', type=str, help='Override mode')
    args = parser.parse_args()

    # 1. Load Configuration
    cm = ConfigManager(args.config)
    mode = args.mode if args.mode else cm.get('mode')
    device = cm.get('device', 'cuda')
    
    # 2. Initialize Viz Manager
    viz = VisualizationManager(cm)

    # 3. Data Setup
    data_cfg = cm.get('data_details')
    train_loader, val_loader, test_loader, class_weights = make_loaders(
        data_cfg['path'], 
        strategy=cm.get('augment'), 
        batch_size=cm.get('training')['batch_size'],
        split_file=data_cfg.get('split_file', 'auto'),
    )

    # 4. Model Setup
    arch_cfg = cm.get('arch_details')
    model = MeatGuardS(num_classes=arch_cfg['num_classes']).to(device)
    
    if mode == 'training':
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = AdamW(model.parameters(), lr=cm.get('training')['lr'], 
                          weight_decay=cm.get('training')['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=cm.get('training')['epochs'], eta_min=1e-6)
        
        trainer = MeatGuardTrainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device=device)
        
        history = {'train_loss': [], 'val_acc': []}
        for epoch in range(cm.get('training')['epochs']):
            loss = trainer.train_epoch()
            _, acc = trainer.validate()
            scheduler.step()
            history['train_loss'].append(loss)
            history['val_acc'].append(acc)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
        
        # Auto-Visualize Training Results
        viz.plot_metric_curve(range(len(history['train_loss'])), history['train_loss'], 
                              title="Training Loss", ylabel="Loss", filename="train_loss.png")
        viz.plot_metric_curve(range(len(history['val_acc'])), history['val_acc'], 
                              title="Validation Accuracy", ylabel="Accuracy", filename="val_acc.png")

    elif mode == 'export':
        # [TFLite export logic from utils.export]
        pass

    elif mode == 'inference':
        # [Inference + XAI logic]
        pass

if __name__ == "__main__":
    main()
