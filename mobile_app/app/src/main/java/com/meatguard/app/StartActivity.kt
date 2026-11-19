package com.meatguard.app

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.meatguard.app.databinding.ActivityStartBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.TensorFlowLite

class StartActivity : AppCompatActivity() {

    private lateinit var binding: ActivityStartBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ReportStorage.init(this)

        binding = ActivityStartBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.title = getString(R.string.app_name)

        binding.bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_saved_reports -> startActivity(Intent(this, SavedReportsActivity::class.java))
                R.id.nav_contact_support -> showSupportDialog()
                R.id.nav_about_us -> showAboutDialog()
            }
            true
        }
        binding.buttonOpenGallery.setOnClickListener {
            startActivity(Intent(this, AnalysisActivity::class.java))
        }
        // Hidden developer tools (Requirements 6 and 7): long press on the title.
        binding.textViewTitle.setOnLongClickListener {
            startActivity(Intent(this, DeveloperActivity::class.java))
            true
        }
    }

    private fun showSupportDialog() {
        AlertDialog.Builder(this)
            .setTitle("How to use MeatGuard")
            .setMessage(
                "1. Tap \"Analyse Meat\", then take a photo or choose one from the gallery.\n\n" +
                "2. Draw the square tightly around the meat, with a little space around it. " +
                "Avoid trays, bags and other objects.\n\n" +
                "3. Use good, even light and keep the whole piece in view.\n\n" +
                "4. Past results are under \"Saved Reports\".\n\n" + ResultText.DISCLAIMER
            )
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showAboutDialog() {
        lifecycleScope.launch {
            val sha = withContext(Dispatchers.IO) {
                try { ModelInfo.sha256(applicationContext) } catch (e: Exception) { "unreadable" }
            }
            val check = if (sha == ModelInfo.EXPECTED_SHA256) "matches the published model" else "DOES NOT MATCH"
            val tflite = try { TensorFlowLite.runtimeVersion() } catch (e: Throwable) { "unknown" }
            AlertDialog.Builder(this@StartActivity)
                .setTitle("About MeatGuard")
                .setMessage(
                    "MeatGuard classifies photographs of raw beef as fresh or spoiled from their " +
                    "visual appearance. All analysis runs on this phone; no photo or result leaves it.\n\n" +
                    "Model: ${ModelInfo.NAME}, ${ModelInfo.VERSION}\n" +
                    "Model file SHA-256 ($check):\n$sha\n\n" +
                    "Pipeline: ${ModelInfo.PIPELINE}\n" +
                    "TensorFlow Lite runtime: $tflite\n" +
                    "App version: ${BuildConfig.VERSION_NAME}\n\n" +
                    "Developed in the MeatGuard II research project.\n\n" + ResultText.DISCLAIMER
                )
                .setPositiveButton("OK", null)
                .show()
        }
    }
}
