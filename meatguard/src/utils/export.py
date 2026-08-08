import os
import shutil
import tempfile
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import DataLoader
import onnx
from onnx2tf import convert

def export_to_tflite(
    pytorch_model,
    calibration_loader,
    model_path="model_int8.tflite",
    input_shape=(1, 3, 224, 224),
):
    """
    Converts a PyTorch model to INT8 TFLite.

    Pipeline:
        PyTorch
            -> ONNX
            -> TensorFlow SavedModel (via onnx2tf)
            -> INT8 TFLite
    """

    pytorch_model.eval().cpu()

    with tempfile.TemporaryDirectory() as tmpdir:

        onnx_path = os.path.join(tmpdir, "model.onnx")
        saved_model_dir = os.path.join(tmpdir, "saved_model")

        # --------------------------------------------------
        # 1. PyTorch -> ONNX
        # --------------------------------------------------

        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            opset_version=18,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
        )

        print(f"[ONNX] Saved: {onnx_path}")

        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # --------------------------------------------------
        # 2. ONNX -> TensorFlow SavedModel
        # --------------------------------------------------

        convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
        )

        print(f"[TF] SavedModel created: {saved_model_dir}")

        # --------------------------------------------------
        # 3. Representative dataset for INT8 calibration
        # --------------------------------------------------

        def representative_dataset():

            for images, _ in calibration_loader:

                if isinstance(images, torch.Tensor):
                    images = images.cpu().numpy()

                images = images.astype("float32")

                # Use a few calibration samples from each batch
                for img in images[:1]:
                    yield [img[None, ...]]

        # --------------------------------------------------
        # 4. SavedModel -> INT8 TFLite
        # --------------------------------------------------

        converter = tf.lite.TFLiteConverter.from_saved_model(
            saved_model_dir
        )

        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]

        converter.representative_dataset = (
            representative_dataset
        )

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]

        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        with open(model_path, "wb") as f:
            f.write(tflite_model)

        print(f"[TFLite] Saved INT8 model: {model_path}")