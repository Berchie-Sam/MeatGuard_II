package com.meatguard.app

import android.os.Bundle
import android.view.View
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.meatguard.app.databinding.ActivityDeveloperBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Hidden developer screen (Requirements 6 and 7). Opened by a long press on the title of the
 * start screen. Input and output files are exchanged with adb (see the folder text on screen).
 */
class DeveloperActivity : AppCompatActivity() {

    private lateinit var binding: ActivityDeveloperBinding
    private lateinit var tools: DevTools
    @Volatile private var cancel = false
    private var running = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDeveloperBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.title = "Developer tools"
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        binding.toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }

        tools = DevTools(applicationContext, ::log) { cancel }
        binding.foldersText.text =
            "Data lives in app-internal storage (${tools.filesRoot}).\n" +
            "Copy it in and out with adb run-as + tar (debug build), see MeatGuard_App_Device_Runs.txt:\n" +
            "  eval/crops/   (mode a)\n" +
            "  eval/photos/  + eval/masks/ (mode b, same name, .png)\n" +
            "  eval/camera/  (end-to-end, optional)\n" +
            "  results/      (output CSV files)\n\n" +
            "Now: ${DevTools.listImages(tools.cropsDir).size} crops, ${DevTools.listImages(tools.photosDir).size} photos, " +
            "${DevTools.listImages(tools.masksDir).size} masks, ${DevTools.listImages(tools.cameraDir).size} camera photos.\n" +
            "Before measuring: charge > 50%, close other apps, let the device cool down, " +
            "screen stays on during a run."

        binding.buttonLatency.setOnClickListener {
            val accels = selectedAccels()
            if (accels.isEmpty()) { log("Select at least one execution setting."); return@setOnClickListener }
            // Read the fields here, on the main thread; the task runs in the background.
            val warmup = number(binding.editWarmup, 20)
            val runs = number(binding.editRuns, 200)
            val pause = number(binding.editPause, 30)
            launchTask("Latency benchmark") { tools.latency(accels, warmup, runs, pause) }
        }
        binding.buttonEndtoend.setOnClickListener {
            val photos = number(binding.editPhotos, 50)
            launchTask("End-to-end timing") { tools.endToEnd(Accel.CPU4, photos) }
        }
        binding.buttonBatchA.setOnClickListener {
            val save = binding.checkSaveCrops.isChecked
            launchTask("Batch (a)") { tools.batch(DevTools.Mode.MODEL_ONLY, Accel.CPU4, save) }
        }
        binding.buttonBatchB.setOnClickListener {
            val save = binding.checkSaveCrops.isChecked
            launchTask("Batch (b)") { tools.batch(DevTools.Mode.GIVEN_MASK, Accel.CPU4, save) }
        }
        binding.buttonCancel.setOnClickListener {
            cancel = true
            log("Stopping after the current step...")
        }
    }

    private fun selectedAccels(): List<Accel> = buildList {
        if (binding.checkCpu1.isChecked) add(Accel.CPU1)
        if (binding.checkCpu4.isChecked) add(Accel.CPU4)
        if (binding.checkNnapi.isChecked) add(Accel.NNAPI)
        if (binding.checkGpu.isChecked) add(Accel.GPU)
    }

    private fun number(v: android.widget.EditText, default: Int): Int =
        v.text.toString().trim().toIntOrNull()?.takeIf { it >= 0 } ?: default

    private fun launchTask(name: String, block: () -> Unit) {
        if (running) return
        running = true
        cancel = false
        setRunning(true)
        log("---- $name ----")
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.Default) { block() }
                log("---- $name finished ----")
            } catch (e: Throwable) {
                log("ERROR: ${e.message ?: e}")
            } finally {
                running = false
                setRunning(false)
            }
        }
    }

    private fun setRunning(on: Boolean) {
        if (on) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        listOf(binding.buttonLatency, binding.buttonEndtoend, binding.buttonBatchA, binding.buttonBatchB)
            .forEach { it.isEnabled = !on }
        binding.buttonCancel.isEnabled = on
    }

    /** Thread-safe: appends a line on the main thread and scrolls down. Always queued (post, not
     *  runOnUiThread), so lines logged on the main thread cannot overtake queued background lines. */
    private fun log(line: String) {
        binding.logText.post {
            binding.logText.append(line + "\n")
            binding.scroll.post { binding.scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    override fun onDestroy() {
        cancel = true
        super.onDestroy()
    }
}
