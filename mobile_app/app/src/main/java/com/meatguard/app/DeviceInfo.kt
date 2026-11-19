package com.meatguard.app

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import org.tensorflow.lite.TensorFlowLite
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Device and run facts written at the top of every results file (Requirements 6.5, 6.7). */
object DeviceInfo {

    fun collect(context: Context): LinkedHashMap<String, String> {
        val m = LinkedHashMap<String, String>()
        m["timestamp"] = SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.US).format(Date())
        m["manufacturer"] = Build.MANUFACTURER
        m["model"] = Build.MODEL
        m["device"] = Build.DEVICE
        m["board"] = Build.BOARD
        m["hardware"] = Build.HARDWARE
        if (Build.VERSION.SDK_INT >= 31) {
            m["soc_manufacturer"] = Build.SOC_MANUFACTURER
            m["soc_model"] = Build.SOC_MODEL
        }
        m["abis"] = Build.SUPPORTED_ABIS.joinToString(" ")
        m["cpu_cores"] = Runtime.getRuntime().availableProcessors().toString()
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        m["ram_total_mb"] = (mi.totalMem / (1024 * 1024)).toString()
        m["ram_available_mb"] = (mi.availMem / (1024 * 1024)).toString()
        m["app_heap_limit_mb"] = am.largeMemoryClass.toString()
        m["android_version"] = Build.VERSION.RELEASE
        m["sdk_int"] = Build.VERSION.SDK_INT.toString()
        m["build_fingerprint"] = Build.FINGERPRINT
        m["app_version"] = BuildConfig.VERSION_NAME
        m["apk_size_bytes"] = File(context.applicationInfo.sourceDir).length().toString()
        m["model_file"] = ModelInfo.FILE
        m["model_size_bytes"] = context.assets.openFd(ModelInfo.FILE).use { it.length }.toString()
        m["model_sha256"] = ModelInfo.sha256(context)
        m["tflite_runtime"] = try { TensorFlowLite.runtimeVersion() } catch (e: Throwable) { "unknown" }
        battery(context)?.let { (pct, charging) ->
            m["battery_percent"] = pct.toString()
            m["charging"] = charging.toString()
        }
        m["thermal_status"] = thermalStatus(context)
        return m
    }

    /** Battery level in percent and whether it is charging, from the sticky battery broadcast. */
    fun battery(context: Context): Pair<Int, Boolean>? {
        val i = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return null
        val level = i.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = i.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        val status = i.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        val charging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL
        return if (level >= 0 && scale > 0) Pair(level * 100 / scale, charging) else null
    }

    /** 0 = none ... 6 = shutdown (PowerManager.THERMAL_STATUS_*); "n/a" before Android 10. */
    fun thermalStatus(context: Context): String {
        if (Build.VERSION.SDK_INT < 29) return "n/a"
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.currentThermalStatus.toString()
    }

    /** Proportional set size of this process in kB (Debug.getMemoryInfo). */
    fun pssKb(): Int = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }.totalPss
}
