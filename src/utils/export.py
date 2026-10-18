"""
Export a (pruned) MeatGuard-S model to ONNX and to a full-integer (INT8) TFLite model, and run the
TFLite model on a data loader so its accuracy can be compared with the PyTorch model.

Pipeline: PyTorch -> ONNX -> TensorFlow SavedModel (onnx2tf) -> INT8 TFLite (post-training quantisation).

onnx2tf converts the network to channels-last, so the TFLite input is (1, 224, 224, 3); calibration and
inference transpose the NCHW tensors from the loaders accordingly (the layout is read from the model).
"""
import os

import numpy as np
import torch


def export_onnx(model, onnx_path, input_shape=(1, 3, 224, 224)):
    """PyTorch -> ONNX (opset 18, dynamic batch), checked with onnx.checker."""
    import onnx
    import inspect
    model = model.eval().cpu()
    # Newer PyTorch defaults to the TorchDynamo exporter; the TorchScript exporter is the one onnx2tf is tested with
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, torch.randn(*input_shape), onnx_path, opset_version=18,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, **extra)
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"[ONNX] {onnx_path}")
    return onnx_path


def _channels_last(input_shape):
    """True when a 4-D model input is (N, H, W, C)."""
    return len(input_shape) == 4 and input_shape[-1] == 3 and input_shape[1] != 3


def _calibration_images(loader, max_images):
    n = 0
    for images, _ in loader:
        for img in images.cpu().numpy().astype("float32"):
            yield img
            n += 1
            if n >= max_images:
                return


def export_tflite_int8(model, calibration_loader, out_dir, name="meatguard_s", max_calibration=500,
                       input_shape=(1, 3, 224, 224)):
    """
    Full-integer post-training quantisation: weights, activations, input and output are int8.

    `calibration_loader` should yield *training* images with the evaluation transform (no augmentation).
    Writes <out_dir>/<name>.onnx, <out_dir>/saved_model/ and <out_dir>/<name>_int8.tflite; returns their paths.
    """
    import tensorflow as tf
    from onnx2tf import convert

    os.makedirs(out_dir, exist_ok=True)
    onnx_path = export_onnx(model, os.path.join(out_dir, f"{name}.onnx"), input_shape)
    saved_model_dir = os.path.join(out_dir, "saved_model")
    # onnx2tf 2.x defaults to the flatbuffer_direct backend, which writes .tflite files but no SavedModel;
    # the TensorFlow converter backend writes the saved_model.pb that the INT8 conversion below loads.
    # onnx2tf.convert is a (*args, **kwargs) wrapper, so the option is tried rather than looked up;
    # onnx2tf 1.x has no such option and always writes a SavedModel.
    kwargs = dict(input_onnx_file_path=onnx_path, output_folder_path=saved_model_dir, non_verbose=True)
    try:
        convert(**kwargs, tflite_backend="tf_converter")
    except TypeError:
        convert(**kwargs)
    if not os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")):
        raise FileNotFoundError(f"onnx2tf wrote no saved_model.pb in {saved_model_dir}")
    print(f"[TF] SavedModel: {saved_model_dir}")

    # Read the converted input layout from the float32 TFLite file that onnx2tf writes alongside
    float_tflite = [f for f in os.listdir(saved_model_dir) if f.endswith("float32.tflite")]
    if float_tflite:
        interp = tf.lite.Interpreter(model_path=os.path.join(saved_model_dir, float_tflite[0]))
        nhwc = _channels_last(tuple(interp.get_input_details()[0]["shape"]))
    else:
        nhwc = True                                     # onnx2tf default
    print(f"[TF] input layout: {'NHWC' if nhwc else 'NCHW'}")

    def representative_dataset():
        for img in _calibration_images(calibration_loader, max_calibration):
            yield [(img.transpose(1, 2, 0) if nhwc else img)[None, ...]]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_path = os.path.join(out_dir, f"{name}_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(converter.convert())
    print(f"[TFLite] INT8 model: {tflite_path}")
    return {"onnx": onnx_path, "saved_model": saved_model_dir, "tflite_int8": tflite_path,
            "input_layout": "NHWC" if nhwc else "NCHW", "n_calibration_images": max_calibration}


def tflite_predict(tflite_path, loader):
    """Run a TFLite model (float or int8 I/O) over a loader; returns (labels, softmax probabilities)."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    nhwc = _channels_last(tuple(inp["shape"]))
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    ys, ps = [], []
    for images, labels in loader:
        for img, y in zip(images.cpu().numpy().astype("float32"), labels.numpy()):
            x = (img.transpose(1, 2, 0) if nhwc else img)[None, ...]
            if inp["dtype"] != np.float32:              # quantise the input with the model's own parameters
                info = np.iinfo(inp["dtype"])
                x = np.clip(np.round(x / in_scale + in_zero), info.min, info.max).astype(inp["dtype"])
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            logits = interp.get_tensor(out["index"]).astype(np.float32)
            if out["dtype"] != np.float32:
                logits = (logits - out_zero) * out_scale
            logits = logits.reshape(-1)
            e = np.exp(logits - logits.max())
            ps.append(e / e.sum())
            ys.append(int(y))
    return np.asarray(ys), np.stack(ps)


def export_to_tflite(pytorch_model, calibration_loader, model_path="model_int8.tflite",
                     input_shape=(1, 3, 224, 224)):
    """Backward-compatible wrapper: write the INT8 TFLite model to `model_path`."""
    out_dir = os.path.dirname(os.path.abspath(model_path))
    name = os.path.splitext(os.path.basename(model_path))[0].replace("_int8", "")
    paths = export_tflite_int8(pytorch_model, calibration_loader, out_dir, name=name, input_shape=input_shape)
    if os.path.abspath(paths["tflite_int8"]) != os.path.abspath(model_path):
        os.replace(paths["tflite_int8"], model_path)
    return model_path
