# Implementation Plan - Enhanced Scan & Detection

This plan outlines the steps to fix camera issues, redesign the `scan.dart` UI to match the provided mockup, and implement the TFLite-based freshness detection in `detect.dart`.

## User Review Required

> [!IMPORTANT]
> To run TFLite models, we need to add the `tflite_v2` package to `pubspec.yaml`. Please ensure you run `flutter pub get` after the file is updated.

> [!NOTE]
> The current "Unknown error initializing camera" might be due to missing permissions or lifecycle issues. I will add explicit permission handling and robust initialization.

## Proposed Changes

### Configuration

#### [MODIFY] [pubspec.yaml](file:///C:/elpi_dev/meatguard/meatguard/meatguard/pubspec.yaml)
- Add `tflite_v2: ^1.0.0` or a similar stable TFLite package.
- Ensure `assets` includes the model files (already listed but double-check paths).

#### [MODIFY] [AndroidManifest.xml](file:///C:/elpi_dev/meatguard/meatguard/meatguard/android/app/src/main/AndroidManifest.xml)
- Add `READ_EXTERNAL_STORAGE` and `READ_MEDIA_IMAGES` (for Android 13+) to support gallery uploads.

---

### UI & Camera Logic

#### [MODIFY] [scan.dart](file:///C:/elpi_dev/meatguard/meatguard/meatguard/lib/scan.dart)
- **Camera Initialization**:
    - Wrap `availableCameras` and `initialize` in a more resilient way.
    - Check for permissions explicitly using `camera`'s built-in capability or `permission_handler`.
- **UI Redesign**:
    - Implement the "Scan Meat" app bar with a help button.
    - Create a custom camera preview with rounded corners and the specific overlay (green frame, crosshair, corner markers).
    - Add the flash toggle (Auto/On/Off) and settings button over the preview.
    - Add zoom controls (0.5x, 1x, 2x).
    - Redesign the bottom control bar with Gallery, Shutter, and Flip buttons.
    - Add the "Tips for best results" card at the bottom.
- **Functionality**:
    - Implement `_toggleFlashMode` to cycle through Auto/On/Off.
    - Implement `_flipCamera` to switch between front and back cameras.
    - Implement `_setZoom` for the zoom buttons.

---

### Detection Logic

#### [MODIFY] [detect.dart](file:///C:/elpi_dev/meatguard/meatguard/meatguard/lib/detect.dart) [FROM EMPTY]
- Implement `DetectScreen` widget.
- Load the TFLite model (`meatguard_s_float16.tflite` or `meatguard_s_float32.tflite`) in `initState`.
- Preprocess the image (resize to model input size).
- Run inference and display results (Fresh/Spoiled/Unknown).
- Show confidence score.
- Provide a "Back to Scan" or "Save Result" option.

## Verification Plan

### Automated Tests
- I will check the syntax of the modified files.
- Note: I cannot run the app directly, so I rely on standard Flutter practices.

### Manual Verification
- User should run the app on a physical device (camera won't work properly on many emulators).
- Verify that the camera starts without the "Unknown error".
- Test gallery upload.
- Test torch toggle.
- Verify that navigating to the detection screen shows a result from the TFLite model.
