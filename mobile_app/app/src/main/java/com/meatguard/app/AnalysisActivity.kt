package com.meatguard.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.canhub.cropper.CropImageContract
import com.canhub.cropper.CropImageContractOptions
import com.canhub.cropper.CropImageOptions
import com.canhub.cropper.CropImageView
import com.meatguard.app.databinding.ActivityAnalysisBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

/**
 * Photo (camera or gallery) -> square crop by the user -> MeatGuard-S -> result.
 * Everything runs on the device; the app has no internet permission.
 */
class AnalysisActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAnalysisBinding
    private var classifier: Classifier? = null
    private var cameraFile: File? = null

    // ---- launchers -------------------------------------------------------------------------

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) startCrop(uri)
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val f = cameraFile
        if (ok && f != null && f.length() > 0) {
            startCrop(FileProvider.getUriForFile(this, "$packageName.fileprovider", f))
        }
    }

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) launchCamera()
        else Toast.makeText(this, "Camera permission is needed to take a photo", Toast.LENGTH_SHORT).show()
    }

    private val cropImage = registerForActivityResult(CropImageContract()) { result ->
        if (result.isSuccessful) {
            result.uriContent?.let { analyse(it) }
        } else if (result.error != null) {
            showError("The crop failed: ${result.error?.message}")
        }
        // Cancelled by the user: nothing to do.
    }

    // ---- lifecycle -------------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAnalysisBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.title = "Meat Freshness Analysis"
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        binding.toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }

        savedInstanceState?.getString(KEY_CAMERA_FILE)?.let { cameraFile = File(it) }
        ReportStorage.init(this)

        binding.placeholderText.text = ResultText.FRAMING_TIP
        binding.disclaimerText.text = ResultText.DISCLAIMER
        setButtonsEnabled(false)
        binding.statusText.text = "Loading model..."

        lifecycleScope.launch {
            try {
                classifier = withContext(Dispatchers.Default) { Classifier(applicationContext) }
                binding.statusText.text = ""
                setButtonsEnabled(true)
                if (classifier?.modelSha256 != ModelInfo.EXPECTED_SHA256) {
                    showError("Warning: the bundled model file is not the expected one (checksum mismatch).")
                }
            } catch (e: Exception) {
                showError("The model could not be loaded: ${e.message}")
            }
        }

        binding.selectImageButton.setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        binding.takePhotoButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                launchCamera()
            } else {
                cameraPermission.launch(Manifest.permission.CAMERA)
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        cameraFile?.let { outState.putString(KEY_CAMERA_FILE, it.absolutePath) }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
        classifier = null
    }

    // ---- steps -----------------------------------------------------------------------------

    private fun launchCamera() {
        val dir = File(cacheDir, "camera").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }          // keep only the latest photo
        val f = File(dir, "photo_${System.currentTimeMillis()}.jpg")
        cameraFile = f
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", f)
        takePicture.launch(uri)
    }

    /** Step 2 (interim): the user draws a square around the meat. */
    private fun startCrop(source: Uri) {
        val options = CropImageOptions(
            fixAspectRatio = true,
            aspectRatioX = 1,
            aspectRatioY = 1,
            guidelines = CropImageView.Guidelines.ON,
            activityTitle = "Frame the meat",
            outputCompressFormat = Bitmap.CompressFormat.PNG,   // lossless: no extra JPEG step
            imageSourceIncludeCamera = false,
            imageSourceIncludeGallery = false
        )
        cropImage.launch(CropImageContractOptions(source, options))
    }

    private fun analyse(cropUri: Uri) {
        val clf = classifier ?: return
        setButtonsEnabled(false)
        binding.statusText.text = "Analysing..."
        binding.resultCard.visibility = View.GONE

        lifecycleScope.launch {
            try {
                val (prediction, report, display) = withContext(Dispatchers.Default) {
                    // The cropper already applied the EXIF orientation; decodeUpright is a no-op then.
                    val crop = ImagePipeline.decodeUpright(contentResolver, cropUri)
                    val px = ImagePipeline.toModelPixels(crop)
                    val p = clf.classify(px)

                    val id = UUID.randomUUID().toString()
                    val dir = ReportStorage.reportDir(this@AnalysisActivity)
                    val display = ImagePipeline.displayCopy(crop)
                    val displayFile = File(dir, "$id.jpg")
                    val modelFile = File(dir, "${id}_224.png")
                    ImagePipeline.save(display, displayFile, Bitmap.CompressFormat.JPEG, 90)
                    ImagePipeline.save(
                        ImagePipeline.pixelsToBitmap(px, ModelInfo.INPUT_SIZE, ModelInfo.INPUT_SIZE),
                        modelFile, Bitmap.CompressFormat.PNG, 100
                    )
                    val r = Report(
                        id = id,
                        timestamp = System.currentTimeMillis(),
                        label = if (p.isSpoiled) "spoiled" else "fresh",
                        pSpoiled = p.pSpoiled,
                        inferenceMicros = p.inferenceMicros,
                        displayImagePath = displayFile.absolutePath,
                        modelImagePath = modelFile.absolutePath,
                        modelSha256 = clf.modelSha256,
                        pipeline = ModelInfo.PIPELINE,
                        appVersion = BuildConfig.VERSION_NAME
                    )
                    ReportStorage.add(this@AnalysisActivity, r)
                    Triple(p, r, BitmapFactory.decodeFile(displayFile.absolutePath))
                }
                showResult(prediction, report, display)
            } catch (e: Exception) {
                showError("The photo could not be analysed: ${e.message}")
            } finally {
                setButtonsEnabled(true)
            }
        }
    }

    private fun showResult(p: Prediction, r: Report, display: Bitmap?) {
        binding.placeholderText.visibility = View.GONE
        binding.photoImageView.setImageBitmap(display)
        binding.statusText.text = ResultText.inferenceTime(p.inferenceMicros)
        binding.resultTitle.text = ResultText.title(r.label, p.confidence)
        binding.resultTitle.setTextColor(
            ContextCompat.getColor(this, if (p.isSpoiled) R.color.result_spoiled else R.color.result_fresh)
        )
        binding.resultExplanation.text = ResultText.explanation(p.isSpoiled, p.isUncertain)
        binding.resultDetails.text =
            "P(spoiled) = ${"%.3f".format(p.pSpoiled)}\n${ModelInfo.NAME}, ${ModelInfo.VERSION}\nPipeline: ${r.pipeline}"
        binding.resultCard.visibility = View.VISIBLE
    }

    private fun showError(msg: String) {
        binding.statusText.text = msg
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        binding.selectImageButton.isEnabled = enabled
        binding.takePhotoButton.isEnabled = enabled
    }

    companion object {
        private const val KEY_CAMERA_FILE = "camera_file"
    }
}
