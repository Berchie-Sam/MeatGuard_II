# MeatGuard II — Source Code (`src/`)

Core Python package: data protocol, MeatGuard-S and its teachers, distillation and compression, explainability,
evaluation and paper figures. Run everything from the repository root as modules (`python -m src.<name>`).

## Pipeline scripts

| Script | Commands | Writes |
|---|---|---|
| [`data/sessions.py`](data/sessions.py) | build capture sessions and the session-level split | `sessions.csv` |
| [`data/segment_meat.py`](data/segment_meat.py) | Grounding DINO + SAM 2.1 meat masks and crops (resumable) | `masks/`, `masked/`, `crops/`, `segmentation.csv` |
| [`train_teacher.py`](train_teacher.py) | `train --arch <name>` (teachers and baselines), `select` | `runs/teacher/<arch>/`, `teacher_selection.json`; `runs/baselines/<arch>/` |
| [`train_student.py`](train_student.py) | `distill` (`--no-kd` for the ablation), `compress`, `export` | `runs/student/{distilled,scratch,pruned,export}/` |
| [`evaluate.py`](evaluate.py) | test evaluation of every trained model | `runs/results/test_results.{json,csv}`, `predictions.csv` |
| [`evaluate_xai.py`](evaluate_xai.py) | Grad-CAM layer selection (validation), faithfulness, meat share, sanity check (test sample) | `runs/xai/` |
| [`collect_results.py`](collect_results.py) | zip all result files (no weights) | `results_bundle.zip` |

Every training command saves `best` and `last` checkpoints, `history.csv` (one row per epoch) and `summary.json`
(configuration, parameters and MACs, best validation metrics, split-file SHA-256, GPU and package versions), and
resumes from `last.pt` when re-run. `--smoke` runs three batches per epoch to check the setup. Model selection uses
validation macro-F1 only; the test split is read by `train_teacher select` (selected teacher only) and `evaluate.py`.

## Packages

| Module | Contents |
|---|---|
| [`data/`](data/) | `sessions.py` (dHash session linking, union-find, stratified session split), `segment_meat.py` (meat masks and crops; `load_cropped_mask` returns a mask in the frame of its crop), `transforms.py` (augmentation strategies; `domain` is used for training), `dataset.py` (`make_loaders`: session split from `sessions.csv`, class weights from the training split) |
| [`model/`](model/) | `architecture.py` (MeatGuard-S, 1.14 M parameters), `teacher.py` (ImageNet teachers `efficientnet_b0`, `convnext_tiny`, `swin_t`, `resnet50` and baselines `mobilenet_v3_small`, `shufflenet_v2_x1_0`; `load_teacher`), `distillation.py` (KD and feature-alignment losses), `kd_trainer.py` (`KDTrainer`: loss of Eq. 5 with a linear adaptor from the pooled Stage-3 output to the teacher's penultimate features), `compression.py` (`MeatGuardPruner`: dependency-aware L1 channel pruning with Torch-Pruning), `trainer.py` (simple supervised trainer, not used by the pipeline) |
| [`xai/`](xai/) | `gradcam.py`, `occlusion.py` (mean probability drop per pixel), `shap_module.py` (GradientExplainer averaged over SLIC superpixels), `faithfulness.py` (deletion / insertion curves and AUC, random-order baseline, meat share, cascading randomisation sanity check) |
| [`utils/`](utils/) | `metrics.py` (accuracy, macro-F1, spoiled recall, ROC-AUC, session-bootstrap CIs, McNemar test), `export.py` (ONNX and full-integer INT8 TFLite export; `tflite_predict` runs a TFLite model on a loader), `config_manager.py`, `viz_manager.py` |
| [`viz/`](viz/) | paper figures: `pipeline_figure.py` (Figure 1), `arch_figure.py` (Figure 2, VisualTorch render checked against the model), `flowcharts.py` |
| [`dashboard/`](dashboard/) | Streamlit research dashboard |
| [`main.py`](main.py) | config-driven runner: `python -m src.main --mode <step>\|all` calls the scripts above with the settings in `configs/config.yaml` (`--smoke`, `--dry-run`, `--runs`); used by the `Makefile` |

## Usage from Python

```python
from src.data.dataset import make_loaders
from src.model.architecture import MeatGuardS
from src.train_student import load_student

train_loader, val_loader, test_loader, class_weights = make_loaders(
    "data/processed/full/crops", strategy="domain", split_file="auto")   # finds data/processed/full/sessions.csv
model = MeatGuardS(num_classes=2)
pruned = load_student("runs/student/pruned/best_model.pt")            # whole pruned module
```

Class indices: `Fresh_CowMeat = 0`, `Spoiled_CowMeat = 1` (spoiled is the positive class). Inputs are 224 × 224 RGB
normalised with the ImageNet mean and standard deviation.

## Dependencies

See [`requirements.txt`](../requirements.txt). Figures use a separate environment (`.venv-viz`, with VisualTorch).
