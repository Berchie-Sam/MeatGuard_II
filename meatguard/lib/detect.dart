// lib/detect.dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Result object returned after analysis.
class DetectionResult {
  final bool isFresh;
  final String status; // e.g. "FRESH", "SPOILED"
  final double confidence; // 0.0 – 1.0
  final String? sampleName;
  final String? modelVersion;

  DetectionResult({
    required this.isFresh,
    required this.status,
    required this.confidence,
    this.sampleName,
    this.modelVersion,
  });
}

/// Analyzes the given image file using the TFLite model.
Future<DetectionResult> analyzeImage(File imageFile) async {
  final interpreter = await _getInterpreter();

  final inputTensors = interpreter.getInputTensors();
  final outputTensors = interpreter.getOutputTensors();

  if (inputTensors.isEmpty || outputTensors.isEmpty) {
    throw Exception('Invalid TFLite model: No tensors found.');
  }

  final inputTensor = inputTensors.first;
  final outputTensor = outputTensors.first;

  // 1. Preprocess into flat TypedData
  final inputBuffer = await _preprocessImage(imageFile, inputTensor);

  // 2. Prepare output buffer
  final outputShape = outputTensor.shape;
  final outputType = outputTensor.type;
  final outputSize = outputShape.reduce((a, b) => a * b);

  Object outputBuffer;
  if (outputType == TensorType.int8) {
    outputBuffer = Int8List(outputSize);
  } else {
    outputBuffer = Float32List(outputSize);
  }

  // 3. Run inference
  // tflite_flutter's run method handles flat buffers that match tensor size.
  interpreter.run(inputBuffer, outputBuffer);

  // 4. Parse result
  return _parseOutput(outputBuffer, outputTensor);
}

Interpreter? _interpreter;

Future<Interpreter> _getInterpreter() async {
  if (_interpreter != null) return _interpreter!;
  const modelPath = 'lib/assets/models/meatguard_s_float16.tflite';
  final modelData = await rootBundle.load(modelPath);
  final options = InterpreterOptions()..threads = 4..useNnApiForAndroid = true;
  _interpreter = await Interpreter.fromBuffer(modelData.buffer.asUint8List(), options: options);
  return _interpreter!;
}

Future<TypedData> _preprocessImage(File imageFile, Tensor inputTensor) async {
  final bytes = await imageFile.readAsBytes();
  final image = img.decodeImage(bytes);
  if (image == null) throw Exception('Failed to decode image');

  final shape = inputTensor.shape; 
  final type = inputTensor.type;

  int h, w, c;
  bool isNCHW = false;

  if (shape.length == 4) {
    if (shape[1] == 3) {
      isNCHW = true;
      c = shape[1]; h = shape[2]; w = shape[3];
    } else {
      h = shape[1]; w = shape[2]; c = shape[3];
    }
  } else {
    throw Exception('Unsupported model input shape: $shape');
  }

  final resized = img.copyResize(image, width: w, height: h);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  if (type == TensorType.int8) {
    final scale = inputTensor.params.scale;
    final zeroPoint = inputTensor.params.zeroPoint;
    final buffer = Int8List(h * w * c);

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final pixel = resized.getPixel(x, y);
        final r = ((pixel.r / 255.0 - mean[0]) / std[0]);
        final g = ((pixel.g / 255.0 - mean[1]) / std[1]);
        final b = ((pixel.b / 255.0 - mean[2]) / std[2]);

        int qr = (r / scale + zeroPoint).round().clamp(-128, 127);
        int qg = (g / scale + zeroPoint).round().clamp(-128, 127);
        int qb = (b / scale + zeroPoint).round().clamp(-128, 127);

        if (isNCHW) {
          buffer[0 * h * w + y * w + x] = qr;
          buffer[1 * h * w + y * w + x] = qg;
          buffer[2 * h * w + y * w + x] = qb;
        } else {
          int base = (y * w + x) * 3;
          buffer[base + 0] = qr;
          buffer[base + 1] = qg;
          buffer[base + 2] = qb;
        }
      }
    }
    return buffer;
  } else {
    final buffer = Float32List(h * w * c);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final pixel = resized.getPixel(x, y);
        final r = ((pixel.r / 255.0 - mean[0]) / std[0]);
        final g = ((pixel.g / 255.0 - mean[1]) / std[1]);
        final b = ((pixel.b / 255.0 - mean[2]) / std[2]);

        if (isNCHW) {
          buffer[0 * h * w + y * w + x] = r;
          buffer[1 * h * w + y * w + x] = g;
          buffer[2 * h * w + y * w + x] = b;
        } else {
          int base = (y * w + x) * 3;
          buffer[base + 0] = r;
          buffer[base + 1] = g;
          buffer[base + 2] = b;
        }
      }
    }
    return buffer;
  }
}

DetectionResult _parseOutput(Object outputBuffer, Tensor outputTensor) {
  List<double> logits;

  if (outputBuffer is Int8List) {
    final scale = outputTensor.params.scale;
    final zeroPoint = outputTensor.params.zeroPoint;
    logits = outputBuffer.map((q) => (q - zeroPoint) * scale).toList();
  } else if (outputBuffer is Float32List) {
    logits = outputBuffer.toList();
  } else {
    throw Exception('Unsupported output buffer type');
  }

  // The output might have a batch dimension if it was converted from Float32List.
  // MeatGuard-S output is [1, 2].
  final probs = _softmax(logits);
  final probFresh = probs[0];
  final probSpoiled = probs[1];

  final isFresh = probFresh > probSpoiled;
  return DetectionResult(
    isFresh: isFresh,
    status: isFresh ? 'FRESH' : 'SPOILED',
    confidence: isFresh ? probFresh : probSpoiled,
    sampleName: 'Beef Sample',
    modelVersion: 'MeatGuard-S',
  );
}

List<double> _softmax(List<double> logits) {
  final maxLogit = logits.reduce((a, b) => a > b ? a : b);
  final exp = logits.map((x) => math.exp(x - maxLogit)).toList();
  final sumExp = exp.reduce((a, b) => a + b);
  return exp.map((x) => x / sumExp).toList();
}
