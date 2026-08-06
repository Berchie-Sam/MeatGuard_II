import 'dart:io';
import 'package:flutter/material.dart';
import 'package:tflite_v2/tflite_v2.dart';

class DetectScreen extends StatefulWidget {
  final String imagePath;
  const DetectScreen({super.key, required this.imagePath});

  @override
  State<DetectScreen> createState() => _DetectScreenState();
}

class _DetectScreenState extends State<DetectScreen> {
  bool _loading = true;
  List? _outputs;
  String _freshnessStatus = "Analyzing...";
  double _confidence = 0.0;

  @override
  void initState() {
    super.initState();
    _loadModel().then((value) {
      _classifyImage(File(widget.imagePath));
    });
  }

  Future<void> _loadModel() async {
    try {
      await Tflite.loadModel(
        model: "lib/assets/models/meatguard_s_float16.tflite",
        // labels: "lib/assets/models/labels.txt", // Optional if labels exist
        numThreads: 1,
        isAsset: true,
        useGpuDelegate: false,
      );
    } catch (e) {
      debugPrint("Error loading model: $e");
    }
  }

  Future<void> _classifyImage(File image) async {
    try {
      var output = await Tflite.runModelOnImage(
        path: image.path,
        numResults: 2,
        threshold: 0.2,
        imageMean: 127.5,
        imageStd: 127.5,
      );

      if (mounted) {
        setState(() {
          _loading = false;
          _outputs = output;
          if (_outputs != null && _outputs!.isNotEmpty) {
            // Assuming the model returns labels in order or uses index
            // If labels file is missing, we might need to interpret the index
            var result = _outputs![0];
            _confidence = (result['confidence'] ?? 0.0) * 100;
            int labelIndex = result['index'] ?? 0;
            
            // Map index to human readable status (adjust based on model)
            if (labelIndex == 0) {
              _freshnessStatus = "Fresh Beef";
            } else {
              _freshnessStatus = "Spoiled / Non-fresh";
            }
          } else {
            _freshnessStatus = "Unable to determine";
          }
        });
      }
    } catch (e) {
      debugPrint("Error during inference: $e");
      if (mounted) {
        setState(() {
          _loading = false;
          _freshnessStatus = "Error during analysis";
        });
      }
    }
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      appBar: AppBar(
        title: const Text('Detection Result', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: _loading
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: Color(0xFFC62828)),
                  SizedBox(height: 20),
                  Text("Processing with AI Model...", style: TextStyle(color: Colors.grey)),
                ],
              ),
            )
          : SingleChildScrollView(
              child: Column(
                children: [
                  const SizedBox(height: 20),
                  // Image Preview
                  Container(
                    margin: const EdgeInsets.symmetric(horizontal: 24),
                    height: 300,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 10, offset: const Offset(0, 5)),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: Image.file(File(widget.imagePath), fit: BoxFit.cover),
                    ),
                  ),
                  const SizedBox(height: 40),
                  
                  // Result Card
                  Container(
                    margin: const EdgeInsets.symmetric(horizontal: 24),
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                      boxShadow: [
                        BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 20, offset: const Offset(0, 10)),
                      ],
                    ),
                    child: Column(
                      children: [
                        const Text(
                          "FRESHNESS STATUS",
                          style: TextStyle(color: Colors.grey, letterSpacing: 1.5, fontSize: 12, fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          _freshnessStatus,
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: _freshnessStatus.contains("Fresh") ? Colors.green : Colors.red,
                          ),
                        ),
                        const SizedBox(height: 24),
                        const Divider(),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text("Confidence", style: TextStyle(color: Colors.black54)),
                            Text("${_confidence.toStringAsFixed(1)}%", style: const TextStyle(fontWeight: FontWeight.bold)),
                          ],
                        ),
                        const SizedBox(height: 12),
                        LinearProgressIndicator(
                          value: _confidence / 100,
                          backgroundColor: Colors.grey.shade200,
                          color: _freshnessStatus.contains("Fresh") ? Colors.green : Colors.red,
                          borderRadius: BorderRadius.circular(10),
                          minHeight: 8,
                        ),
                      ],
                    ),
                  ),
                  
                  const SizedBox(height: 40),
                  
                  // Action Buttons
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 24),
                    child: Row(
                      children: [
                        Expanded(
                          child: OutlinedButton(
                            onPressed: () => Navigator.pop(context),
                            style: OutlinedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                            ),
                            child: const Text("Scan Again"),
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: ElevatedButton(
                            onPressed: () {
                              // Save to history logic would go here
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(content: Text("Result saved to history")),
                              );
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFFC62828),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                            ),
                            child: const Text("Save Result"),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 40),
                ],
              ),
            ),
    );
  }
}
