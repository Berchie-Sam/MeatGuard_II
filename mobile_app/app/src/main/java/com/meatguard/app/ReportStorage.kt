package com.meatguard.app

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File

/** One analysis, as stored on the device (nothing leaves the phone). */
data class Report(
    val id: String,
    val timestamp: Long,
    val label: String,            // "fresh" or "spoiled"
    val pSpoiled: Float,
    val inferenceMicros: Long,
    val displayImagePath: String?,  // crop for display (JPEG, <= 512 px)
    val modelImagePath: String?,    // the exact 224 x 224 input of the model (PNG)
    val modelSha256: String,
    val pipeline: String,
    val appVersion: String
) {
    val isSpoiled: Boolean get() = label == "spoiled"
    val confidence: Float get() = if (isSpoiled) pSpoiled else 1f - pSpoiled
}

object ReportStorage {
    private const val PREFS = "meatguard_reports"
    private const val KEY = "reports_v1"
    private val gson = Gson()
    private val reports = mutableListOf<Report>()
    private var loaded = false

    fun reportDir(context: Context) = File(context.filesDir, "reports")

    @Synchronized
    fun init(context: Context) {
        if (loaded) return
        val json = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, null)
        if (json != null) {
            val type = object : TypeToken<List<Report>>() {}.type
            reports.clear()
            reports.addAll(gson.fromJson<List<Report>>(json, type))
        }
        loaded = true
    }

    @Synchronized
    fun all(): List<Report> = reports.toList()

    @Synchronized
    fun add(context: Context, report: Report) {
        reports.add(0, report)
        save(context)
    }

    @Synchronized
    fun delete(context: Context, id: String) {
        reports.filter { it.id == id }.forEach { r ->
            r.displayImagePath?.let { File(it).delete() }
            r.modelImagePath?.let { File(it).delete() }
        }
        reports.removeAll { it.id == id }
        save(context)
    }

    private fun save(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString(KEY, gson.toJson(reports)).apply()
    }
}
