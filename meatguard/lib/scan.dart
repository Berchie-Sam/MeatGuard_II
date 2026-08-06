import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

import 'history.dart';
import 'reports.dart';
import 'settings.dart';
import 'detect.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> with WidgetsBindingObserver {
  // Camera variables
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  FlashMode _flashMode = FlashMode.auto;
  String? _cameraInitError;
  int _selectedCameraIndex = 0;
  double _currentZoomLevel = 1.0;
  double _minZoomLevel = 1.0;
  double _maxZoomLevel = 1.0;

  // Image picker for gallery
  final ImagePicker _imagePicker = ImagePicker();

  @override
  void initState() {
    debugPrint('Entering initState');
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }

  // Initialize the camera system
  Future<void> _initializeCamera() async {
    debugPrint('Calling _initializeCamera');
    try {
      debugPrint('Requesting camera permission');
      final status = await Permission.camera.request();
      if (!status.isGranted) {
        debugPrint('Camera permission denied: $status');
        if (mounted) {
          setState(() {
            _cameraInitError = status.isPermanentlyDenied
                ? 'Camera permission is permanently denied. Please enable it in settings.'
                : 'Camera permission was denied.';
            _isCameraInitialized = false;
          });
        }
        return;
      }

      debugPrint('Before availableCameras()');
      _cameras = await availableCameras();
      debugPrint('After availableCameras()');

      if (_cameras != null && _cameras!.isNotEmpty) {
        final camera = _cameras![_selectedCameraIndex];

        _cameraController = CameraController(
          camera,
          ResolutionPreset.high,
          enableAudio: false,
          imageFormatGroup: Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
        );

        debugPrint('Before cameraController.initialize()');
        await _cameraController!.initialize();
        debugPrint('After cameraController.initialize()');

        _minZoomLevel = await _cameraController!.getMinZoomLevel();
        _maxZoomLevel = await _cameraController!.getMaxZoomLevel();

        if (mounted) {
          setState(() {
            _isCameraInitialized = true;
            _cameraInitError = null;
          });
        }
      } else {
        debugPrint('No cameras found');
        if (mounted) {
          setState(() {
            _cameraInitError = "No cameras found.";
            _isCameraInitialized = false;
          });
        }
      }
    } on CameraException catch (e) {
      debugPrint('CameraException in _initializeCamera: ${e.code} - ${e.description}');
      if (mounted) {
        setState(() {
          _cameraInitError = 'Camera Error: ${e.description}';
          _isCameraInitialized = false;
        });
      }
    } catch (e) {
      debugPrint('Generic exception in _initializeCamera: $e');
      if (mounted) {
        setState(() {
          _cameraInitError = "Error: $e";
          _isCameraInitialized = false;
        });
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _cameraController;
    if (cameraController == null || !cameraController.value.isInitialized) return;

    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _toggleFlashMode() async {
    if (_cameraController == null || !_isCameraInitialized) return;

    FlashMode nextMode;
    switch (_flashMode) {
      case FlashMode.auto:
        nextMode = FlashMode.always;
        break;
      case FlashMode.always:
        nextMode = FlashMode.off;
        break;
      case FlashMode.off:
        nextMode = FlashMode.auto;
        break;
      default:
        nextMode = FlashMode.auto;
    }

    try {
      await _cameraController!.setFlashMode(nextMode);
      setState(() => _flashMode = nextMode);
    } catch (e) {
      debugPrint("Error setting flash mode: $e");
    }
  }

  Future<void> _flipCamera() async {
    if (_cameras == null || _cameras!.length < 2) return;
    _selectedCameraIndex = (_selectedCameraIndex + 1) % _cameras!.length;
    await _cameraController?.dispose();
    _initializeCamera();
  }

  Future<void> _setZoom(double zoom) async {
    if (_cameraController == null || !_isCameraInitialized) return;
    if (zoom < _minZoomLevel || zoom > _maxZoomLevel) return;
    try {
      await _cameraController!.setZoomLevel(zoom);
      setState(() => _currentZoomLevel = zoom);
    } catch (e) {
      debugPrint("Error setting zoom: $e");
    }
  }

  // Snap photo using the camera
  Future<void> _captureImage() async {
    if (_cameraController == null || !_isCameraInitialized) return;

    try {
      HapticFeedback.mediumImpact();
      final XFile image = await _cameraController!.takePicture();
      _navigateToDetectScreen(image.path);
    } catch (e) {
      debugPrint("Error capturing image: $e");
    }
  }

  // Pick an image from the photo gallery
  Future<void> _pickImageFromGallery() async {
    debugPrint('Entering _pickImageFromGallery');
    try {
      final XFile? image = await _imagePicker.pickImage(source: ImageSource.gallery);
      debugPrint('After _imagePicker.pickImage: ${image?.path}');
      if (image != null) {
        _navigateToDetectScreen(image.path);
      }
    } catch (e) {
      debugPrint("Error picking image from gallery: $e");
    }
  }

  void _navigateToDetectScreen(String imagePath) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DetectScreen(imagePath: imagePath),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black87),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Scan Meat',
          style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 20),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.black87),
            tooltip: 'Retry Camera',
            onPressed: () {
              setState(() {
                _isCameraInitialized = false;
                _cameraInitError = null;
              });
              _initializeCamera();
            },
          ),
          IconButton(
            icon: const Icon(Icons.help_outline, color: Colors.black87),
            onPressed: () {},
          ),
        ],
      ),
      body: Column(
        children: [
          const Text(
            'Position the meat within the frame',
            style: TextStyle(color: Colors.black54, fontSize: 14),
          ),
          const SizedBox(height: 20),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20.0),
              child: Stack(
                children: [
                  // Camera Preview Container
                  ClipRRect(
                    borderRadius: BorderRadius.circular(30),
                    child: Container(
                      color: Colors.black,
                      child: _buildCameraPreview(),
                    ),
                  ),
                  
                  // Flash & Settings Top Overlay
                  if (_isCameraInitialized)
                  Positioned(
                    top: 20,
                    left: 20,
                    right: 20,
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _buildFlashToggleButton(),
                        _buildSettingsButton(),
                      ],
                    ),
                  ),

                  // Scanning Overlay (Frame, Crosshair, Corners)
                  if (_isCameraInitialized)
                  Positioned.fill(
                    child: IgnorePointer(
                      child: CustomPaint(
                        painter: MeatScannerPainter(),
                      ),
                    ),
                  ),

                  // Zoom Controls
                  if (_isCameraInitialized)
                  Positioned(
                    bottom: 30,
                    left: 0,
                    right: 0,
                    child: _buildZoomControls(),
                  ),
                ],
              ),
            ),
          ),

          // Bottom Controls (Gallery, Shutter, Flip)
          _buildBottomControls(),

          // Tips Card
          _buildTipsCard(),
          const SizedBox(height: 10),
        ],
      ),
      bottomNavigationBar: _buildBottomNavigationBar(context),
    );
  }

  Widget _buildCameraPreview() {
    if (_cameraInitError != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.videocam_off, color: Colors.white, size: 50),
            const SizedBox(height: 20),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Text(
                _cameraInitError!,
                style: const TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                setState(() {
                  _isCameraInitialized = false;
                  _cameraInitError = null;
                });
                _initializeCamera();
              },
              child: const Text('Retry Connection'),
            ),
          ],
        ),
      );
    }

    if (!_isCameraInitialized || _cameraController == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: Colors.white),
            SizedBox(height: 16),
            Text('Initializing Camera...', style: TextStyle(color: Colors.white70)),
          ],
        ),
      );
    }

    final size = MediaQuery.of(context).size;
    var scale = size.aspectRatio * _cameraController!.value.aspectRatio;
    if (scale < 1) scale = 1 / scale;

    return Transform.scale(
      scale: scale,
      child: Center(
        child: CameraPreview(_cameraController!),
      ),
    );
  }

  Widget _buildFlashToggleButton() {
    IconData icon;
    String label;
    switch (_flashMode) {
      case FlashMode.auto:
        icon = Icons.flash_auto;
        label = 'Auto';
        break;
      case FlashMode.always:
        icon = Icons.flash_on;
        label = 'On';
        break;
      case FlashMode.off:
        icon = Icons.flash_off;
        label = 'Off';
        break;
      default:
        icon = Icons.flash_auto;
        label = 'Auto';
    }

    return GestureDetector(
      onTap: _toggleFlashMode,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.9),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 18, color: Colors.black87),
            const SizedBox(width: 4),
            Text(label, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
          ],
        ),
      ),
    );
  }

  Widget _buildSettingsButton() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.9),
        shape: BoxShape.circle,
      ),
      child: IconButton(
        icon: const Icon(Icons.settings, size: 20, color: Colors.black87),
        onPressed: () {},
      ),
    );
  }

  Widget _buildZoomControls() {
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.9),
          borderRadius: BorderRadius.circular(25),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildZoomButton(0.5, '0.5'),
            const SizedBox(width: 8),
            _buildZoomButton(1.0, '1x'),
            const SizedBox(width: 8),
            _buildZoomButton(2.0, '2'),
          ],
        ),
      ),
    );
  }

  Widget _buildZoomButton(double zoom, String label) {
    bool isSelected = _currentZoomLevel == zoom;
    return GestureDetector(
      onTap: () => _setZoom(zoom),
      child: Container(
        width: 40,
        height: 40,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: isSelected ? Colors.grey.shade300 : Colors.transparent,
          shape: BoxShape.circle,
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.black : Colors.grey,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  Widget _buildBottomControls() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24.0, horizontal: 40.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _buildIconButton(Icons.photo_library_outlined, 'Gallery', _pickImageFromGallery),
          _buildShutterButton(),
          _buildIconButton(Icons.flip_camera_android, 'Flip', _flipCamera),
        ],
      ),
    );
  }

  Widget _buildIconButton(IconData icon, String label, VoidCallback onTap) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          decoration: BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(color: Colors.black12, blurRadius: 4, offset: Offset(0, 2)),
            ],
          ),
          child: IconButton(
            icon: Icon(icon, color: Colors.black87),
            onPressed: onTap,
          ),
        ),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(fontSize: 12, color: Colors.black54)),
      ],
    );
  }

  Widget _buildShutterButton() {
    return GestureDetector(
      onTap: _captureImage,
      child: Container(
        width: 80,
        height: 80,
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.grey.shade300, width: 4),
        ),
        child: Container(
          decoration: const BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(color: Colors.black12, blurRadius: 4, offset: Offset(0, 2)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTipsCard() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFBE9E7), // Very light red
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        children: [
          const Icon(Icons.lightbulb, color: Colors.redAccent),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Tips for best results',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                ),
                Text(
                  'Use good lighting. Hold the camera steady. Ensure the meat is clearly visible.',
                  style: TextStyle(fontSize: 12, color: Colors.black54),
                ),
              ],
            ),
          ),
          Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey.shade400),
        ],
      ),
    );
  }

  Widget _buildBottomNavigationBar(BuildContext context) {
    return BottomNavigationBar(
      type: BottomNavigationBarType.fixed,
      selectedItemColor: Colors.red.shade700,
      unselectedItemColor: Colors.grey,
      currentIndex: 1,
      items: const [
        BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
        BottomNavigationBarItem(icon: Icon(Icons.document_scanner_outlined), label: 'Scan'),
        BottomNavigationBarItem(icon: Icon(Icons.bar_chart), label: 'Reports'),
        BottomNavigationBarItem(icon: Icon(Icons.history), label: 'History'),
        BottomNavigationBarItem(icon: Icon(Icons.settings_outlined), label: 'Settings'),
      ],
      onTap: (index) {
        if (index == 0) {
          Navigator.popUntil(context, (route) => route.isFirst);
        } else if (index == 2) {
          Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const ReportsScreen()));
        } else if (index == 3) {
          Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const HistoryScreen()));
        } else if (index == 4) {
          Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const SettingsScreen()));
        }
      },
    );
  }
}

class MeatScannerPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;

    final double padding = 40.0;
    final double cornerSize = 25.0;

    // Outer Corners
    final Rect rect = Rect.fromLTRB(padding, padding, size.width - padding, size.height - padding);
    
    // TL
    canvas.drawLine(Offset(rect.left, rect.top), Offset(rect.left + cornerSize, rect.top), paint);
    canvas.drawLine(Offset(rect.left, rect.top), Offset(rect.left, rect.top + cornerSize), paint);
    // TR
    canvas.drawLine(Offset(rect.right, rect.top), Offset(rect.right - cornerSize, rect.top), paint);
    canvas.drawLine(Offset(rect.right, rect.top), Offset(rect.right, rect.top + cornerSize), paint);
    // BL
    canvas.drawLine(Offset(rect.left, rect.bottom), Offset(rect.left + cornerSize, rect.bottom), paint);
    canvas.drawLine(Offset(rect.left, rect.bottom), Offset(rect.left, rect.bottom - cornerSize), paint);
    // BR
    canvas.drawLine(Offset(rect.right, rect.bottom), Offset(rect.right - cornerSize, rect.bottom), paint);
    canvas.drawLine(Offset(rect.right, rect.bottom), Offset(rect.right, rect.bottom - cornerSize), paint);

    // Inner Green Box
    final greenPaint = Paint()
      ..color = Colors.green.withOpacity(0.8)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
    
    final innerRectSize = size.width * 0.45;
    final innerRect = Rect.fromCenter(
      center: Offset(size.width / 2, size.height / 2),
      width: innerRectSize,
      height: innerRectSize * 0.7,
    );
    
    canvas.drawRRect(RRect.fromRectAndRadius(innerRect, const Radius.circular(12)), greenPaint);

    // Crosshair
    canvas.drawLine(
      Offset(size.width / 2 - 10, size.height / 2),
      Offset(size.width / 2 + 10, size.height / 2),
      greenPaint,
    );
    canvas.drawLine(
      Offset(size.width / 2, size.height / 2 - 10),
      Offset(size.width / 2, size.height / 2 + 10),
      greenPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
