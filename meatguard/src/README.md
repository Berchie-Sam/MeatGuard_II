# MeatGuard II — Source Code (`src/`)

Core Python package implementing the MeatGuard-S model development, explainability, and evaluation pipeline.

## Package Structure

| Module | Description |
|---|---|
| [`data/`](data/) | Data ingestion, augmentation transforms (6 strategies), dataset loaders — strict parity with Project 1 benchmark |
| [`model/`](model/) | MeatGuard-S hybrid architecture, training loop, knowledge distillation, structured pruning |
| [`xai/`](xai/) | Grad-CAM, SHAP (GradientExplainer + SLIC), Occlusion Sensitivity, faithfulness metrics (Insertion/Deletion AUC, IoU) |
| [`utils/`](utils/) | Configuration manager (YAML resolution), visualization manager (Matplotlib/Plotly, dashboard/report modes), multi-format model export |
| [`dashboard/`](dashboard/) | Streamlit research dashboard for real-time metric monitoring |
| [`main.py`](main.py) | CLI entry point driven by `configs/config.yaml` and the Makefile |

## Usage

```python
# Inside the MeatGuard_II root
from src.data.dataset import make_loaders
from src.model.architecture import MeatGuardS

train_loader, val_loader, test_loader, cw = make_loaders("path/to/data")
model = MeatGuardS(num_classes=2)
```

## Dependencies

See [`requirements.txt`](../requirements.txt) or [`environment.yml`](../environment.yml) at the project root.
