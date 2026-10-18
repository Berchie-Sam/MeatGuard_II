# MeatGuard App

## Model

The app uses `meatguard_s_int8_mixed.tflite` (0.95 MB): 8-bit integer weights and activations, with the two layer
normalisations and the attention softmax kept in 32-bit float. It is built by `python -m src.quantize_variants
--attn-pattern layer_normalization,softmax` (variant `int8_float_attn`) from the SavedModel that
`python -m src.train_student export` writes to `runs/student/export/`, and copied to
`runs/student/export/meatguard_s_int8_mixed.tflite`. `summary.json` next to the variant gives its validation
metrics and the operations kept in float.

The full-integer model `meatguard_s_int8.tflite` is **not** used: it classifies too many spoiled samples as fresh
(test spoiled recall 0.820 against 0.950 for the mixed model).

| Item | Value |
|---|---|
| Input | 1 × 224 × 224 × 3, **float32**, channels last (NHWC) |
| Preprocessing | resize to 224 × 224 RGB, scale to [0, 1], normalise with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) |
| Output | 1 × 2 logits, **float32**; apply softmax |
| Classes | index 0 = Fresh, index 1 = Spoiled |

The model quantises and dequantises internally; the app passes and receives float tensors and needs no
quantisation parameters.

**Framing.** The model was trained on images cropped to the meat, with the background replaced by the mean colour
(RGB 124, 116, 104). Photos in the app should be framed tightly on the meat (for example with an on-screen guide
square) so that the input resembles the training crops.

**Latency.** Report the median of repeated timed inferences after a few warm-up runs, per device and per delegate
(CPU / XNNPACK, GPU, NNAPI), together with the model file size. The float layer-normalisation and softmax operations
may not be supported by every delegate; check which operations fall back to the CPU (for example with the TFLite
benchmark tool's `--enable_op_profiling=true`) and report the delegate that actually runs the model.
