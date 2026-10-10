# MeatGuard II — Runbook

How to run the full experiment, in order, in three equivalent ways:

1. **Makefile** — one command; skips steps whose results already exist.
2. **Python scripts** — one command per step; full control over every option.
3. **Notebooks** — Google Colab with the codebase and data on Google Drive.

All three run the same code (`src/`) with the same settings (`configs/config.yaml`) and write the same files under
`runs/`. Paths are relative to the repository root.

---

## 0. Before you start

### Environment

```bash
pip install -r requirements.txt
```

A CUDA GPU is needed for training in reasonable time. Data preparation (step 1) runs on a CPU.

### Data layout

| Folder | Contents | Produced by |
|---|---|---|
| `data/raw/MeatScan/` | original images, `Fresh_CowMeat/` and `Spoiled_CowMeat/` | download from [Zenodo](https://doi.org/10.5281/zenodo.16764338) |
| `data/processed/full/` | `sessions.csv`, `masks/`, `masked/`, `crops/`, `segmentation.csv` | step 1 |

Both folders are git-ignored. If the data live elsewhere (for example on another drive), either change the paths in
`configs/data/meatscan_crops.yaml`, or link the folders into the repository (Windows, from the repository root):

```
mklink /J data\raw\MeatScan   E:\Datasets\MeatScan
mklink /J data\processed\full E:\Datasets\full
```

Do not upload the repository to Google Drive while it contains `data/`; the dataset would be uploaded with it.

### Order of the steps

| # | Step | Needs | Writes | Notes |
|---|---|---|---|---|
| 1a | `sessions` | raw images | `data/processed/full/sessions.csv` | once; capture sessions + split |
| 1b | `segment` | raw images | `data/processed/full/{masks,masked,crops}/` | once; resumable; hours on CPU |
| 2 | `teacher` | 1 | `runs/teacher/<arch>/`, `teacher_selection.json` | 4 candidates, then selection |
| 3 | `baselines` | 1 (and 2, to reuse EfficientNet-B0) | `runs/baselines/<arch>/` | comparison models |
| 4 | `scratch` | 1 | `runs/student/scratch/` | MeatGuard-S without distillation |
| 5 | `distill` | 2 | `runs/student/distilled/` | MeatGuard-S |
| 6 | `compress` | 5 | `runs/student/pruned/` | pruning + fine-tuning |
| 7 | `export` | 6 | `runs/student/export/` | ONNX + INT8 TFLite (for the app) |
| 8 | `evaluate` | 2–7 | `runs/results/` | test set, all models |
| 9 | `xai` | 6 | `runs/xai/` | evaluation of the explanations |
| 10 | `collect` | any | `runs/results_bundle.zip` | result files to share (no weights) |

Steps 3 and 4 are independent of 5–7 and can run at any point after step 1 (step 3 after step 2 if the
EfficientNet-B0 teacher run should be reused rather than retrained).

Every training step saves `last.pt` after each epoch and **resumes** when re-run; a run that has finished
(including by early stopping) is not trained further. Model selection uses validation data only; the test set is
read only by `teacher` (once, for the selected teacher) and `evaluate`.

---

## 1. Makefile

`make` is not installed on Windows by default (install it with `choco install make` or `winget install GnuWin32.Make`,
or use section 2 instead). On Linux, macOS and Colab it is available.

```bash
make data                 # steps 1a + 1b (once)
make smoke                # whole pipeline with 3 batches per step -> runs_smoke/ (checks the setup in minutes)
make                      # steps 2-10
```

Individual steps (each also runs any earlier step whose result file is missing):

```bash
make teacher
make baselines
make scratch
make distill
make compress
make export
make evaluate
make xai
make collect
```

Options and helpers:

```bash
make dry-run                                  # print every command without running it
make PYTHON=.venv/Scripts/python.exe          # use a specific interpreter
make RUNS=runs_v2                             # write results to another folder
make CONFIG=configs/my_config.yaml            # use another config
make clean                                    # delete runs_smoke/ only (never runs/)
```

Each target is the summary file its step writes last (for example `runs/student/distilled/summary.json`), so
`make` skips finished steps. To redo a step, delete its folder under `runs/`.

---

## 2. Python scripts

### 2.1 Through the runner (settings from `configs/config.yaml`)

`src/main.py` runs one step, or steps 2–10 with `--mode all`:

```bash
python -m src.main --mode sessions
python -m src.main --mode segment
python -m src.main --mode all --smoke         # check the setup -> runs_smoke/
python -m src.main --mode all                 # steps 2-10
python -m src.main --mode distill             # one step
python -m src.main --mode all --dry-run       # print the commands only
python -m src.main --mode all --runs runs_v2  # other output folder
```

Unlike `make`, the runner does not skip finished steps; each training step notices its own finished run and
stops immediately, but `teacher select`, `evaluate`, `xai` and `collect` are recomputed.

### 2.2 Step by step (every option available)

With `DATA=data/processed/full/crops` (`sessions.csv` is found next to it; pass `--split-file` otherwise). Add
`--smoke` to any training command to run three batches only.

```bash
# 1. Data preparation (once)
python -m src.data.sessions     --src data/raw/MeatScan --out data/processed/full/sessions.csv
python -m src.data.segment_meat --src data/raw/MeatScan --out data/processed/full

# 2. Teacher: train the four candidates, then select one on validation macro-F1
python -m src.train_teacher train --arch efficientnet_b0 --data $DATA
python -m src.train_teacher train --arch convnext_tiny   --data $DATA
python -m src.train_teacher train --arch swin_t          --data $DATA
python -m src.train_teacher train --arch resnet50        --data $DATA
python -m src.train_teacher select --data $DATA

# 3. Lightweight baselines (same recipe, separate folder)
python -m src.train_teacher train --arch mobilenet_v3_small --data $DATA --out runs/baselines
python -m src.train_teacher train --arch shufflenet_v2_x1_0 --data $DATA --out runs/baselines
python -m src.train_teacher train --arch efficientnet_b0    --data $DATA --out runs/baselines   # or copy runs/teacher/efficientnet_b0

# 4. MeatGuard-S without distillation (ablation)
python -m src.train_student distill --no-kd --data $DATA

# 5-7. MeatGuard-S: distillation, pruning + fine-tuning, export
python -m src.train_student distill  --data $DATA        # teacher from runs/teacher/teacher_selection.json
python -m src.train_student compress --data $DATA        # reads runs/student/distilled/best.pt
python -m src.train_student export   --data $DATA        # reads runs/student/pruned/best_model.pt

# 8-10. Evaluation and results bundle
python -m src.evaluate     --data $DATA
python -m src.evaluate_xai --data $DATA                  # masks from data/processed/full/masks
python -m src.collect_results --runs runs --out runs/results_bundle.zip
```

On Windows PowerShell, set the variable with `$DATA = "data/processed/full/crops"`.

Frequently used options (see `python -m <module> --help` for all):

| Script | Option | Default |
|---|---|---|
| `train_teacher train` | `--epochs`, `--batch-size`, `--lr`, `--patience`, `--restart`, `--no-pretrained` | 30, 64, 3e-4, 8 |
| `train_student distill` | `--epochs`, `--lr`, `--alpha`, `--temperature`, `--beta`, `--teacher`, `--restart` | 100, 1e-3, 0.3, 4.0, 0.1 |
| `train_student compress` | `--amount`, `--epochs`, `--lr`, `--student` | 0.3, 10, 1e-4 |
| `train_student export` | `--model`, `--calibration-images` | pruned model, 500 |
| `evaluate` | `--runs`, `--no-latency` | `runs` |
| `evaluate_xai` | `--test-per-class` (0 = all), `--shap-background`, `--layer`, `--no-shap` | 100, 50 |
| all | `--device`, `--num-workers`, `--seed`, `--split-file`, `--out` | auto, 4, 42 |

---

## 3. Notebooks (Google Colab)

### Upload

1. Zip `crops/`, `masks/` and `sessions.csv` from `data/processed/full/` (the other folders are not needed) and
   upload the zip to Drive as `MyDrive/MeatGuard/full.zip`.
2. Upload the repository folder, **without** `data/`, to `MyDrive/MeatGuard_II`.
3. Open the notebooks from `MyDrive/MeatGuard_II/notebooks/` with a GPU runtime
   (*Runtime → Change runtime type → GPU*).

If you use other locations, edit `REPO_DIR` and `DATA_SRC` in the first code cell of each notebook.

### Order

| Notebook | Steps | Needs |
|---|---|---|
| `00_teacher_training` | 2 | data zip |
| `01_meatguard_training` | 3, 4 | data zip (reuses the EfficientNet-B0 run from 00 if it exists) |
| `02_distill_compress_export` | 5, 6, 7 | 00 |
| `03_inference_and_xai` | 8 | 00–02 (and 01 for the comparison rows) |
| `04_xai_faithfulness` | 9 | 02 |

Run the cells top to bottom. Each notebook mounts Drive, installs the missing packages, copies the data to the
Colab machine, checks it, runs a smoke test where training is involved, then the full step. Results go to
`MyDrive/MeatGuard_II/runs/`, so they survive disconnects: after a disconnect, re-run the notebook from the top and
training resumes from the last epoch. Every notebook ends by writing `runs/results_bundle.zip`.

---

## 4. After the runs

Send `runs/results_bundle.zip` (JSON and CSV files only; no model weights). The files that feed the paper:

| File | Contents |
|---|---|
| `runs/teacher/teacher_selection.json` | validation ranking of the candidates; test metrics of the selected teacher |
| `runs/*/<arch or step>/summary.json`, `history.csv` | settings, model cost, best epoch and validation metrics, GPU and versions; per-epoch curves |
| `runs/student/pruned/summary.json` | cost before/after pruning, accuracy before/after fine-tuning |
| `runs/student/export/summary.json` | file sizes, INT8 vs FP32 agreement |
| `runs/results/test_results.json`, `.csv`, `predictions.csv` | test metrics with session-bootstrap CIs, McNemar tests, cost, CPU latency; per-image predictions |
| `runs/xai/xai_summary.json`, `xai_per_image.csv`, `sanity_check.csv` | Grad-CAM layer, deletion/insertion AUC, meat share, sanity check |

Not produced by the codebase: latency on phones (measured in the Android app; see `mobile_app/README.md`) and the
details of the test devices.

---

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No sessions.csv found` | `--data` does not point at `data/processed/full/crops`, or step 1a has not run |
| `... is missing images listed in sessions.csv` | the crops are incomplete; finish step 1b (it resumes) or re-copy the data |
| `Checkpoint was trained on a different split file` | `sessions.csv` changed since the run started; use `--restart` or a new `--runs` folder |
| `Run notebook 00 first` / no `teacher_selection.json` | run step 2 (teacher) before step 5 (distill) |
| Out of GPU memory | lower `--batch-size` (teacher candidates: 32; MeatGuard-S: 16) |
| Export fails on `onnx2tf` | install the export packages: `pip install onnx onnxruntime onnxscript onnx2tf onnx_graphsurgeon sng4onnx ai_edge_litert` |
| A step should be redone | delete its folder under `runs/` (or pass `--restart` to a training command) |
