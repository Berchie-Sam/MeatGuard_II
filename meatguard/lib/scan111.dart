import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

// Import your detection logic
import 'detect.dart'; // Make sure this file exports detectFreshness(File image)

// -----------------------------------------------------------------------------
// SCAN SCREEN (scan.dart) - Camera & Gallery integration
// -----------------------------------------------------------------------------
class ScanScreen extends StatefulWidget {
  const ScanScreen({Key? key}) : super(key: key);

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _isCameraInitialized = false;
  bool _isRearCamera = true;
  FlashMode _flashMode = FlashMode.auto;
  double _zoomLevel = 1.0;
  bool _isAnalyzing = false;
  int _currentIndex = 1; // Scan tab

  // Available cameras
  List<CameraDescription>? _cameras;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _controller?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      if (_controller != null) {
        _initCamera();
      }
    }
  }

  Future<void> _initCamera() async {
    // Request camera permission
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) {
      // Show error and fallback to gallery only
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Camera permission denied. Use gallery instead.'),
            backgroundColor: Colors.red,
          ),
        );
      }
      return;
    }

    // Get available cameras
    try {
      _cameras = await availableCameras();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to load cameras: $e')),
        );
      }
      return;
    }

    if (_cameras == null || _cameras!.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No camera found on this device.')),
        );
      }
      return;
    }

    // Select rear camera by default, or the first available
    CameraDescription selectedCamera = _cameras!.firstWhere(
          (cam) => cam.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras!.first,
    );

    if (!_isRearCamera) {
      final front = _cameras!.firstWhere(
            (cam) => cam.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras!.first,
      );
      selectedCamera = front;
    }

    _controller = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    _initializeControllerFuture = _controller!.initialize();
    try {
      await _initializeControllerFuture;
      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Camera initialization error: $e')),
        );
      }
      _isCameraInitialized = false;
    }
  }

  // ---------- Navigation methods ----------
  void _onBottomNavTapped(int index) {
    setState(() {
      _currentIndex = index;
    });
    switch (index) {
      case 0:
        Navigator.pushReplacementNamed(context, '/home');
        break;
      case 1:
        break;
      case 2:
        Navigator.pushReplacementNamed(context, '/reports');
        break;
      case 3:
        Navigator.pushReplacementNamed(context, '/history');
        break;
      case 4:
        Navigator.pushReplacementNamed(context, '/settings');
        break;
    }
  }

  // ---------- Capture/Upload logic ----------
  Future<void> _captureAndAnalyze() async {
    if (!_isCameraInitialized || _controller == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Camera not ready.')),
      );
      return;
    }

    setState(() => _isAnalyzing = true);

    try {
      final XFile picture = await _controller!.takePicture();
      final File imageFile = File(picture.path);
      await _processImage(imageFile);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to capture image: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isAnalyzing = false);
      }
    }
  }

  Future<void> _pickFromGallery() async {
    final picker = ImagePicker();
    final XFile? picked = await picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 85,
    );
    if (picked == null) return;

    // Copy to a temporary file (optional)
    final tempDir = await getTemporaryDirectory();
    final tempPath = path.join(tempDir.path, 'gallery_${DateTime.now().millisecondsSinceEpoch}.jpg');
    final File tempFile = await File(picked.path).copy(tempPath);
    await _processImage(tempFile);
  }

  Future<void> _processImage(File imageFile) async {
    // Show analysis overlay
    setState(() => _isAnalyzing = true);

    // Call detection from detect.dart
    try {
      final result = await detectFreshness(imageFile); // Assume this returns a DetectionResult
      if (!mounted) return;
      // Show result dialog
      _showResultDialog(result);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Detection error: $e')),
      );
    } finally {
      if (mounted) {
        setState(() => _isAnalyzing = false);
      }
    }
  }

  void _showResultDialog(DetectionResult result) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Icon(
              result.isFresh ? Icons.check_circle_rounded : Icons.warning_rounded,
              color: result.isFresh ? const Color(0xFF43A047) : const Color(0xFFE53935),
              size: 28,
            ),
            const SizedBox(width: 8),
            const Text('Freshness Result', style: TextStyle(fontSize: 18)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Sample: ${result.sampleName ?? 'Beef'}',
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
            ),
            const SizedBox(height: 8),
            Text('Status: ${result.status} (${(result.confidence * 100).toStringAsFixed(1)}% Confidence)'),
            const SizedBox(height: 4),
            Text('Model: ${result.modelVersion ?? 'MeatScan v1.2.3'}'),
            const SizedBox(height: 12),
            LinearProgressIndicator(
              value: result.confidence,
              backgroundColor: Colors.grey.shade200,
              color: result.isFresh ? const Color(0xFF43A047) : const Color(0xFFE53935),
              minHeight: 8,
              borderRadius: BorderRadius.circular(4),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close', style: TextStyle(color: Color(0xFFE53935))),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFE53935),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            ),
            onPressed: () {
              Navigator.pop(context);
              Navigator.pushReplacementNamed(context, '/history');
            },
            child: const Text('View History', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  // ---------- UI Build ----------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF9FAFB),
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(context),
            Expanded(
              child: SingleChildScrollView(
                physics: const BouncingScrollPhysics(),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                child: Column(
                  children: [
                    _buildCameraViewfinder(context),
                    const SizedBox(height: 16),
                    _buildCaptureActionRow(context),
                    const SizedBox(height: 16),
                    _buildTipsCard(context),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ),
            _buildBottomNavigationBar(),
          ],
        ),
      ),
    );
  }

  // ----- App Bar -----
  Widget _buildAppBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.black87, size: 20),
            onPressed: () => Navigator.pushReplacementNamed(context, '/home'),
          ),
          const Column(
            children: [
              Text(
                'Scan Meat',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF2D3748),
                ),
              ),
              SizedBox(height: 2),
              Text(
                'Position the meat within the frame',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey,
                ),
              ),
            ],
          ),
          IconButton(
            icon: const Icon(Icons.help_outline_rounded, color: Colors.black87, size: 24),
            onPressed: () {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('Scanning Guide'),
                  content: const Text(
                      'Ensure proper lighting and keep the meat sample steady inside the green viewfinder box for highest TFLite accuracy.'
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('Got it', style: TextStyle(color: Color(0xFFE53935))),
                    ),
                  ],
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // ----- Camera Viewfinder -----
  Widget _buildCameraViewfinder(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 380,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.12),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: Stack(
          fit: StackFit.expand,
          children: [
            // Camera preview or placeholder
            if (_isCameraInitialized && _controller != null)
              CameraPreview(_controller!)
            else
              Container(
                color: const Color(0xFF3E1F1F),
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.camera_alt,
                        color: Colors.white.withOpacity(0.3),
                        size: 60,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        _isCameraInitialized ? 'Camera ready' : 'Initializing camera...',
                        style: TextStyle(color: Colors.white.withOpacity(0.6)),
                      ),
                    ],
                  ),
                ),
              ),

            // Overlay UI elements
            Padding(
              padding: const EdgeInsets.all(20.0),
              child: Stack(
                children: [
                  // Corner brackets
                  Positioned(top: 8, left: 8, child: _buildCornerBracket(true, true)),
                  Positioned(top: 8, right: 8, child: _buildCornerBracket(true, false)),
                  Positioned(bottom: 50, left: 8, child: _buildCornerBracket(false, true)),
                  Positioned(bottom: 50, right: 8, child: _buildCornerBracket(false, false)),

                  // Flash mode selector
                  Positioned(
                    top: 12,
                    left: 12,
                    child: InkWell(
                      onTap: _toggleFlashMode,
                      borderRadius: BorderRadius.circular(20),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.5),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _flashMode == FlashMode.auto
                                  ? Icons.flash_auto_rounded
                                  : _flashMode == FlashMode.torch
                                  ? Icons.flash_on_rounded
                                  : Icons.flash_off_rounded,
                              color: Colors.white,
                              size: 14,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              _flashMode == FlashMode.auto
                                  ? 'Auto'
                                  : _flashMode == FlashMode.torch
                                  ? 'On'
                                  : 'Off',
                              style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),

                  // Settings gear icon
                  Positioned(
                    top: 12,
                    right: 12,
                    child: InkWell(
                      onTap: () => Navigator.pushReplacementNamed(context, '/settings'),
                      borderRadius: BorderRadius.circular(20),
                      child: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.5),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.settings_outlined, color: Colors.white, size: 16),
                      ),
                    ),
                  ),

                  // Central targeting box
                  Center(
                    child: Container(
                      width: 140,
                      height: 100,
                      decoration: BoxDecoration(
                        border: Border.all(color: const Color(0xFF4CAF50), width: 2),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Center(
                        child: Icon(Icons.add, color: Color(0xFF4CAF50), size: 24),
                      ),
                    ),
                  ),

                  // Zoom selector
                  Positioned(
                    bottom: 12,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.6),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [0.5, 1.0, 2.0].map((zoom) {
                            final isSelected = (_zoomLevel - zoom).abs() < 0.01;
                            return GestureDetector(
                              onTap: () {
                                if (_controller != null) {
                                  _controller!.setZoomLevel(zoom);
                                  setState(() => _zoomLevel = zoom);
                                }
                              },
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                decoration: BoxDecoration(
                                  color: isSelected ? Colors.white : Colors.transparent,
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: Text(
                                  zoom == 1.0 ? '1x' : zoom.toStringAsFixed(1),
                                  style: TextStyle(
                                    color: isSelected ? Colors.black : Colors.white70,
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            );
                          }).toList(),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Analyzing overlay
            if (_isAnalyzing)
              Container(
                color: Colors.black.withOpacity(0.4),
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(color: Colors.white),
                      SizedBox(height: 12),
                      Text(
                        'Running TFLite Inference...',
                        style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildCornerBracket(bool isTop, bool isLeft) {
    return Container(
      width: 24,
      height: 24,
      decoration: BoxDecoration(
        border: Border(
          top: isTop ? const BorderSide(color: Colors.white, width: 3) : BorderSide.none,
          bottom: !isTop ? const BorderSide(color: Colors.white, width: 3) : BorderSide.none,
          left: isLeft ? const BorderSide(color: Colors.white, width: 3) : BorderSide.none,
          right: !isLeft ? const BorderSide(color: Colors.white, width: 3) : BorderSide.none,
        ),
      ),
    );
  }

  void _toggleFlashMode() {
    if (_controller == null) return;
    FlashMode newMode;
    switch (_flashMode) {
      case FlashMode.auto:
        newMode = FlashMode.torch;
        break;
      case FlashMode.torch:
        newMode = FlashMode.off;
        break;
      default:
        newMode = FlashMode.auto;
    }
    _controller!.setFlashMode(newMode);
    setState(() => _flashMode = newMode);
  }

  // ----- Action Row (Gallery / Capture / Flip) -----
  Widget _buildCaptureActionRow(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        // Gallery Button
        InkWell(
          onTap: _pickFromGallery,
          borderRadius: BorderRadius.circular(16),
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.grey.shade200),
              boxShadow: [
                BoxShadow(color: Colors.black.withOpacity(0.02), blurRadius: 6),
              ],
            ),
            child: const Icon(Icons.photo_library_outlined, color: Colors.black87, size: 24),
          ),
        ),

        // Shutter Button
        GestureDetector(
          onTap: _captureAndAnalyze,
          child: Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: const Color(0xFFE53935), width: 3),
              color: Colors.white,
              boxShadow: [
                BoxShadow(color: const Color(0xFFE53935).withOpacity(0.2), blurRadius: 10, offset: const Offset(0, 4)),
              ],
            ),
            child: Center(
              child: Container(
                width: 58,
                height: 58,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  color: Color(0xFFE53935),
                ),
                child: const Icon(Icons.camera_alt, color: Colors.white, size: 26),
              ),
            ),
          ),
        ),

        // Flip Camera Button
        InkWell(
          onTap: () {
            setState(() {
              _isRearCamera = !_isRearCamera;
            });
            _initCamera(); // Reinitialize with new camera
          },
          borderRadius: BorderRadius.circular(16),
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.grey.shade200),
              boxShadow: [
                BoxShadow(color: Colors.black.withOpacity(0.02), blurRadius: 6),
              ],
            ),
            child: const Icon(Icons.flip_camera_ios_outlined, color: Colors.black87, size: 24),
          ),
        ),
      ],
    );
  }

  // ----- Tips Card -----
  Widget _buildTipsCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF5F5),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFFEE2E2)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFFFFECEE),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.lightbulb_rounded, color: Color(0xFFE53935), size: 24),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Tips for best results',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF2D3748),
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Use good lighting. Hold the camera steady. Ensure the meat is clearly visible.',
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.grey[700],
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
          const Icon(Icons.chevron_right, color: Colors.grey, size: 20),
        ],
      ),
    );
  }

  // ----- Bottom Navigation Bar -----
  Widget _buildBottomNavigationBar() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildNavItem(0, Icons.home_rounded, 'Home', Colors.grey.shade600),
          _buildNavItem(1, Icons.qr_code_scanner_rounded, 'Scan', const Color(0xFFE53935)),
          _buildNavItem(2, Icons.bar_chart_rounded, 'Reports', Colors.grey.shade600),
          _buildNavItem(3, Icons.access_time_rounded, 'History', Colors.grey.shade600),
          _buildNavItem(4, Icons.settings_rounded, 'Settings', Colors.grey.shade600),
        ],
      ),
    );
  }

  Widget _buildNavItem(int index, IconData icon, String label, Color color) {
    bool isSelected = _currentIndex == index;
    return InkWell(
      onTap: () => _onBottomNavTapped(index),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: isSelected ? const Color(0xFFE53935) : color,
            size: 24,
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              color: isSelected ? const Color(0xFFE53935) : color,
              fontSize: 10,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }
}

// Example placeholder for DetectionResult (define in detect.dart)
class DetectionResult {
  final bool isFresh;
  final String status;
  final double confidence;
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

// Example detection function signature (implement in detect.dart)
Future<DetectionResult> detectFreshness(File image) async {
  // TODO: Replace with actual TFLite inference
  await Future.delayed(const Duration(seconds: 1)); // Simulate work
  return DetectionResult(
    isFresh: true,
    status: 'FRESH',
    confidence: 0.946,
    sampleName: 'Beef Tenderloin Cut',
    modelVersion: 'MeatScan v1.2.3',
  );
}