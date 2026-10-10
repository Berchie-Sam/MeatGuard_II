# MeatGuard App

## Model

The app uses `meatguard_s_int8.tflite`, written by `python -m src.train_student export` (notebook
`02_distill_compress_export`) to `runs/student/export/`. `runs/student/export/summary.json` gives its size and its
agreement with the PyTorch model on validation images.

| Item | Value |
|---|---|
| Input | 1 × 224 × 224 × 3, **int8**, channels last (NHWC) |
| Preprocessing | resize to 224 × 224 RGB, scale to [0, 1], normalise with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), then quantise: `q = round(x / scale + zero_point)` using the input tensor's quantisation parameters |
| Output | 1 × 2 logits, **int8**; dequantise with `(q - zero_point) * scale`, then softmax |
| Classes | index 0 = Fresh, index 1 = Spoiled |

Read `scale` and `zero_point` from the interpreter (`getInputTensor(0).quantizationParams()` and the output
equivalent) rather than hard-coding them; they change whenever the model is re-exported.

**Framing.** The model was trained on images cropped to the meat, with the background replaced by the mean colour
(RGB 124, 116, 104). Photos in the app should be framed tightly on the meat (for example with an on-screen guide
square) so that the input resembles the training crops.

**Latency.** Report the median of repeated timed inferences after a few warm-up runs, per device and per delegate
(CPU / XNNPACK, GPU, NNAPI), together with the model file size.
