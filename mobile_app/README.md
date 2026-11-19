# MeatGuard App (research build)

Native Android app (Kotlin, TensorFlow Lite 2.17) that runs MeatGuard-S on the phone, offline. It was adapted from
the app team's Kotlin project. The build steps (JDK 17, SDK and Gradle on D:, command line only) are in
`MeatGuard_App_Build_Setup_Guide.txt` in the paper folder. In short, in a window opened with
`D:\AndroidDev\android-env.ps1`:

```
.\gradlew.bat testDebugUnitTest *> build_log.txt
.\gradlew.bat assembleDebug *> build_log.txt
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Pipeline: photo (camera or gallery) → EXIF orientation → square crop chosen by the user → resize to 224 × 224 with
an exact port of Pillow's bilinear resize (`PilResize.kt`, the method used to make the training crops) → ImageNet
normalisation → model → softmax. Each report stores the analysed 224 × 224 image, P(spoiled), the model checksum,
the pipeline name (`manual-crop-v1`) and the app version.

## Model

The app bundles `app/src/main/assets/meatguard_s_int8_float_attn.tflite` (1,001,768 bytes, SHA-256
`7f1ae042c3680ba551ae3341e4729b1ae81f867adbe8ba8592e497a1e6a47090`; the app checks it at start-up and shows it in
About). It has 8-bit integer weights and activations, with the two layer normalisations and the attention softmax
kept in 32-bit float. It is the variant `int8_float_attn` built by `python -m src.quantize_variants
--attn-pattern layer_normalization,softmax` from the SavedModel that `python -m src.train_student export` writes to
`runs/student/export/`.

The full-integer model `meatguard_s_int8.tflite` is **not** used: it classifies too many spoiled samples as fresh.

| Item | Value |
|---|---|
| Input | 1 × 224 × 224 × 3, **float32**, channels last (NHWC) |
| Preprocessing | 224 × 224 RGB, scale to [0, 1], normalise with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) |
| Output | 1 × 2 logits, **float32**; apply softmax |
| Classes | index 0 = Fresh, index 1 = Spoiled |

The model quantises and dequantises internally; the app passes and receives float tensors and needs no
quantisation parameters.

**Framing.** The model was trained on images cropped to the meat. The user frames the meat with the square crop
so that the input resembles the training crops; automatic segmentation on the phone is future work.

**Latency.** Time only the model call, after warm-up runs, per device and per delegate (CPU / XNNPACK, GPU,
NNAPI), and report it with the model file size. The float layer-normalisation and softmax operations may not be
supported by every delegate; check which operations fall back to the CPU (for example with the TFLite benchmark
tool's `--enable_op_profiling=true`) and report the delegate that actually runs the model.
