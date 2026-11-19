package com.meatguard.app

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.random.Random

/**
 * Developer tools for the paper (Requirements 6 and 7): latency benchmark, end-to-end timing and
 * batch evaluation. Everything runs one image at a time on the calling (background) thread.
 *
 * Folders (app-internal storage; no storage permission needed):
 *   <files>/eval/crops/   224 x 224 crops for mode (a), any sub-folders (e.g. fresh/, spoiled/)
 *   <files>/eval/photos/  original photos for mode (b) and end-to-end timing
 *   <files>/eval/masks/   one mask PNG per photo, same relative path, extension .png (255 = meat)
 *   <files>/eval/camera/  optional: full-size camera photos for end-to-end timing (no masks;
 *                         the centre square is used, as in the app); used instead of photos/ if present
 *   <files>/results/      CSV files written by the tools
 * where <files> = /data/user/0/com.meatguard.app/files (Context.filesDir).
 * Internal storage, because some devices (e.g. HONOR, Android 14) refuse adb writes to
 * /sdcard/Android/data. Debug builds reach it with "adb exec-in/exec-out run-as <package> tar ...".
 */
class DevTools(private val context: Context, private val log: (String) -> Unit, private val cancelled: () -> Boolean) {

    val filesRoot: File = context.filesDir
    val cropsDir = File(filesRoot, "eval/crops")
    val photosDir = File(filesRoot, "eval/photos")
    val masksDir = File(filesRoot, "eval/masks")
    val cameraDir = File(filesRoot, "eval/camera")
    val resultsDir = File(filesRoot, "results")

    init {
        listOf(cropsDir, photosDir, masksDir, cameraDir, resultsDir).forEach { it.mkdirs() }
    }

    // ---- 6.1-6.3, 6.5: latency of the model call -------------------------------------------

    fun latency(accels: List<Accel>, warmup: Int, runs: Int, pauseSeconds: Int): File {
        val stamp = stamp()
        val rawFile = File(resultsDir, "latency_$stamp.csv")
        val sumFile = File(resultsDir, "latency_${stamp}_summary.csv")
        val device = DeviceInfo.collect(context)
        val (pixels, source) = benchmarkInput()
        log("Input: $source")

        PrintWriter(rawFile).use { raw ->
            PrintWriter(sumFile).use { sum ->
                header(raw, device, "warmup=$warmup runs=$runs pause_s=$pauseSeconds input=$source")
                header(sum, device, "warmup=$warmup runs=$runs pause_s=$pauseSeconds input=$source")
                raw.println("setting,run,wall_us,native_us")
                sum.println((listOf("setting", "status", "load_ms", "pss_before_kb", "pss_loaded_kb", "pss_after_kb",
                    "thermal_start", "thermal_end", "battery_start", "notes") +
                    Summary.HEADER.map { "wall_ms_$it" } + Summary.HEADER.map { "native_ms_$it" }).joinToString(","))

                for ((k, accel) in accels.withIndex()) {
                    if (cancelled()) break
                    if (k > 0 && pauseSeconds > 0) {
                        log("Pause ${pauseSeconds}s (cool-down)...")
                        for (s in 0 until pauseSeconds) { if (cancelled()) break; Thread.sleep(1000) }
                    }
                    log("== ${accel.label}")
                    val thermalStart = DeviceInfo.thermalStatus(context)
                    val battery = DeviceInfo.battery(context)?.first?.toString() ?: ""
                    System.gc()
                    val pssBefore = DeviceInfo.pssKb()
                    val clf = try {
                        Classifier(context, accel)
                    } catch (e: Throwable) {
                        log("   FAILED: ${e.message}")
                        sum.println(csv(listOf(accel.name, "failed", "", pssBefore, "", "", thermalStart, "", battery,
                            e.message ?: e.toString())))
                        continue
                    }
                    try {
                        val pssLoaded = DeviceInfo.pssKb()
                        clf.setInput(pixels)
                        repeat(warmup) { if (!cancelled()) clf.runOnce() }
                        val wall = ArrayList<Double>(runs)
                        val native = ArrayList<Double>(runs)
                        for (i in 0 until runs) {
                            if (cancelled()) break
                            val ns = clf.runOnce()
                            val nat = clf.lastNativeNanos()
                            wall.add(ns / 1e6); if (nat >= 0) native.add(nat / 1e6)
                            raw.println("${accel.name},$i,${ns / 1000},${if (nat >= 0) nat / 1000 else ""}")
                        }
                        val err = clf.delegateErrors()
                        val notes = listOfNotNull(clf.notes.ifEmpty { null }, err).joinToString("; ")
                        val pssAfter = DeviceInfo.pssKb()
                        val status = if (err != null) "errors" else if (wall.size < runs) "cancelled" else "ok"
                        val ws = if (wall.isNotEmpty()) Summary.of(wall) else null
                        val nsum = if (native.isNotEmpty()) Summary.of(native) else null
                        sum.println(csv(listOf(accel.name, status, "%.2f".format(Locale.US, clf.loadMicros / 1000.0), pssBefore,
                            pssLoaded, pssAfter, thermalStart, DeviceInfo.thermalStatus(context), battery, notes) +
                            (ws?.values() ?: List(10) { "" }) + (nsum?.values() ?: List(10) { "" })))
                        log("   load %.1f ms; ".format(clf.loadMicros / 1000.0) + (ws?.line() ?: "no runs"))
                        if (notes.isNotEmpty()) log("   note: $notes")
                    } finally {
                        clf.close()
                    }
                }
            }
        }
        log("Saved ${rawFile.name} and ${sumFile.name}")
        return sumFile
    }

    /** First crop of eval/crops if there is one, otherwise a fixed pseudo-random image. */
    private fun benchmarkInput(): Pair<IntArray, String> {
        listImages(cropsDir).firstOrNull()?.let { f ->
            val bmp = ImagePipeline.decodeFile(f, applyExif = false)
            try { return Pair(ImagePipeline.toModelPixels(bmp), "crop ${f.relativeTo(cropsDir).path}") } finally { bmp.recycle() }
        }
        val rnd = Random(0)
        val n = ModelInfo.INPUT_SIZE * ModelInfo.INPUT_SIZE
        return Pair(IntArray(n) { (0xFF shl 24) or rnd.nextInt(0x1000000) }, "synthetic (seed 0)")
    }

    // ---- 6.4: end-to-end timing per stage -----------------------------------------------------

    fun endToEnd(accel: Accel, count: Int): File {
        // Real camera photos if there are any (realistic decode/resize cost), else the dataset photos
        val dir = if (listImages(cameraDir).isNotEmpty()) cameraDir else photosDir
        val photos = listImages(dir).take(count)
        require(photos.isNotEmpty()) { "No photos in ${dir.absolutePath}" }
        val out = File(resultsDir, "endtoend_${stamp()}.csv")
        val device = DeviceInfo.collect(context)
        val clf = Classifier(context, accel)
        val warm = minOf(3, photos.size - 1)
        log("End-to-end on ${photos.size} photos, ${accel.label} (first $warm marked as warm-up)")
        val totals = ArrayList<Double>()
        try {
            PrintWriter(out).use { w ->
                header(w, device, "setting=${accel.name} photos=${photos.size} folder=${dir.name} warmup_rows=$warm")
                w.println("file,warmup,width,height,mask_used,decode_ms,mask_ms,crop_ms,resize_ms,normalise_ms," +
                          "inference_ms,post_ms,total_ms,p_spoiled")
                for ((i, f) in photos.withIndex()) {
                    if (cancelled()) break
                    val rel = f.relativeTo(dir).path.replace('\\', '/')
                    val maskFile = if (dir == photosDir) maskFor(f) else null
                    var t = System.nanoTime()
                    // Stage 1: decode (+ EXIF orientation when there is no research mask to match)
                    val bmp = ImagePipeline.decodeFile(f, applyExif = maskFile == null)
                    val width = bmp.width; val height = bmp.height
                    val px = IntArray(width * height).also { bmp.getPixels(it, 0, width, 0, 0, width, height) }
                    bmp.recycle()
                    val tDecode = lap(t).also { t = System.nanoTime() }
                    // Stage 2: segmentation - here the given mask is read from file
                    val mask = maskFile?.let { readMask(it, width, height) }
                    val tMask = lap(t).also { t = System.nanoTime() }
                    // Stages 3-4: fill + square crop (or the centre square without a mask)
                    val (square, side) = squareFor(px, mask, width, height)
                    val tCrop = lap(t).also { t = System.nanoTime() }
                    // Stage 5: resize to 224 (Pillow bilinear)
                    val m224 = PilResize.resize(square, side, side, ModelInfo.INPUT_SIZE, ModelInfo.INPUT_SIZE)
                    val tResize = lap(t).also { t = System.nanoTime() }
                    // Stage 6: normalise into the input tensor
                    clf.setInput(m224)
                    val tNorm = lap(t)
                    // Stage 7: inference
                    val tInfer = clf.runOnce() / 1e6
                    t = System.nanoTime()
                    // Stage 8: softmax + label
                    val lg = clf.logits()
                    val p = Prediction.fromLogits(lg[0], lg[1], 0)
                    val tPost = lap(t)
                    val total = tDecode + tMask + tCrop + tResize + tNorm + tInfer + tPost
                    if (i >= warm) totals.add(total)
                    w.println(csv(listOf(rel, if (i < warm) 1 else 0, width, height, if (mask != null) 1 else 0,
                        f2(tDecode), f2(tMask), f2(tCrop), f2(tResize), f2(tNorm), f2(tInfer), f2(tPost), f2(total),
                        "%.6f".format(Locale.US, p.pSpoiled))))
                    if ((i + 1) % 10 == 0) log("   ${i + 1}/${photos.size}")
                }
            }
        } finally {
            clf.close()
        }
        if (totals.isNotEmpty()) log("Total per photo: " + Summary.of(totals).line())
        log("Saved ${out.name}")
        return out
    }

    // ---- 7: batch evaluation --------------------------------------------------------------------

    enum class Mode(val code: String) { MODEL_ONLY("a_model_only"), GIVEN_MASK("b_given_mask") }

    fun batch(mode: Mode, accel: Accel, saveCrops: Boolean): File {
        val dir = if (mode == Mode.MODEL_ONLY) cropsDir else photosDir
        val files = listImages(dir)
        require(files.isNotEmpty()) { "No images in ${dir.absolutePath}" }
        val stamp = stamp()
        val out = File(resultsDir, "batch_${mode.code}_$stamp.csv")
        val cropOut = File(resultsDir, "batch_${mode.code}_${stamp}_crops")
        val device = DeviceInfo.collect(context)
        val clf = Classifier(context, accel)
        log("Batch ${mode.code}: ${files.size} images, ${accel.label}")
        var done = 0; var failed = 0
        try {
            PrintWriter(out).use { w ->
                header(w, device, "mode=${mode.code} setting=${accel.name} images=${files.size} pipeline=" +
                    (if (mode == Mode.MODEL_ONLY) "model-only" else "given-mask (no EXIF rotation, as the Python pipeline)"))
                w.println("file,mode,setting,width,height,resized,mask_area,crop_left,crop_top,crop_side,no_meat," +
                          "logit_fresh,logit_spoiled,p_fresh,p_spoiled,label,inference_ms,error")
                for ((i, f) in files.withIndex()) {
                    if (cancelled()) break
                    val rel = f.relativeTo(dir).path.replace('\\', '/')
                    try {
                        val bmp = ImagePipeline.decodeFile(f, applyExif = false)
                        val width = bmp.width; val height = bmp.height
                        var area = ""; var win: MaskCrop.Window? = null; var noMeat = ""; var resized = 0
                        val m224: IntArray
                        if (mode == Mode.MODEL_ONLY) {
                            resized = if (width == ModelInfo.INPUT_SIZE && height == ModelInfo.INPUT_SIZE) 0 else 1
                            m224 = ImagePipeline.toModelPixels(bmp)
                            bmp.recycle()
                        } else {
                            val px = IntArray(width * height).also { bmp.getPixels(it, 0, width, 0, 0, width, height) }
                            bmp.recycle()
                            val maskFile = maskFor(f) ?: throw IllegalStateException("no mask")
                            val mask = readMask(maskFile, width, height)
                            val a = MaskCrop.area(mask)
                            area = "%.6f".format(Locale.US, a)
                            noMeat = if (a < MaskCrop.SMALL_MASK) "1" else "0"
                            win = MaskCrop.window(mask, width, height) ?: throw IllegalStateException("empty mask")
                            val square = MaskCrop.fillAndCrop(px, mask, width, height, win)
                            m224 = PilResize.resize(square, win.side, win.side, ModelInfo.INPUT_SIZE, ModelInfo.INPUT_SIZE)
                            resized = if (win.side == ModelInfo.INPUT_SIZE) 0 else 1
                        }
                        if (saveCrops) {
                            val cropBmp = ImagePipeline.pixelsToBitmap(m224, ModelInfo.INPUT_SIZE, ModelInfo.INPUT_SIZE)
                            ImagePipeline.save(cropBmp, File(cropOut, rel.substringBeforeLast('.') + ".png"),
                                Bitmap.CompressFormat.PNG, 100)
                            cropBmp.recycle()
                        }
                        val p = clf.classify(m224)
                        w.println(csv(listOf(rel, mode.code, accel.name, width, height, resized, area,
                            win?.left ?: "", win?.top ?: "", win?.side ?: "", noMeat,
                            "%.6f".format(Locale.US, p.logitFresh), "%.6f".format(Locale.US, p.logitSpoiled),
                            "%.6f".format(Locale.US, p.pFresh), "%.6f".format(Locale.US, p.pSpoiled),
                            if (p.isSpoiled) "spoiled" else "fresh", "%.3f".format(Locale.US, p.inferenceMicros / 1000.0), "")))
                        done++
                    } catch (e: Throwable) {
                        failed++
                        w.println(csv(listOf(rel, mode.code, accel.name, "", "", "", "", "", "", "", "",
                            "", "", "", "", "", "", e.message ?: e.toString())))
                    }
                    if ((i + 1) % 100 == 0) { w.flush(); log("   ${i + 1}/${files.size}") }
                }
            }
        } finally {
            clf.close()
        }
        log("Done: $done images, $failed failed. Saved ${out.name}" + if (saveCrops) " and ${cropOut.name}/" else "")
        return out
    }

    // ---- helpers ----------------------------------------------------------------------------------

    /** Mask of a photo: eval/masks/<same relative path>.png, or null if there is none. */
    private fun maskFor(photo: File): File? {
        val rel = photo.relativeTo(photosDir).path
        val f = File(masksDir, rel.substringBeforeLast('.') + ".png")
        return if (f.isFile) f else null
    }

    /** Mask PNG -> booleans (Python: np.asarray(Image.open(p)) > 127; grey PNG, so R = G = B). */
    private fun readMask(f: File, w: Int, h: Int): BooleanArray {
        val bmp = ImagePipeline.decodeFile(f, applyExif = false)
        try {
            require(bmp.width == w && bmp.height == h) { "mask ${bmp.width}x${bmp.height} != photo ${w}x$h" }
            val px = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }
            return BooleanArray(w * h) { ((px[it] shr 16) and 0xFF) > 127 }
        } finally {
            bmp.recycle()
        }
    }

    /** Fill + square crop with a mask, or the centre square without one. Returns (pixels, side). */
    private fun squareFor(px: IntArray, mask: BooleanArray?, w: Int, h: Int): Pair<IntArray, Int> {
        if (mask != null) {
            val win = MaskCrop.window(mask, w, h)
            if (win != null) return Pair(MaskCrop.fillAndCrop(px, mask, w, h, win), win.side)
        }
        val side = minOf(w, h)
        val left = (w - side) / 2; val top = (h - side) / 2
        val sq = IntArray(side * side)
        for (y in 0 until side) System.arraycopy(px, (y + top) * w + left, sq, y * side, side)
        return Pair(sq, side)
    }

    private fun header(w: PrintWriter, device: Map<String, String>, run: String) {
        device.forEach { (k, v) -> w.println("# $k: $v") }
        w.println("# run: $run")
    }

    private fun lap(t0: Long) = (System.nanoTime() - t0) / 1e6
    private fun f2(v: Double) = "%.3f".format(Locale.US, v)
    private fun stamp() = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

    private fun csv(values: List<Any?>): String = values.joinToString(",") { v ->
        val s = when (v) {
            null -> ""
            is Double -> "%.4f".format(Locale.US, v)
            else -> v.toString()
        }
        if (s.contains(',') || s.contains('"') || s.contains('\n')) "\"" + s.replace("\"", "\"\"") + "\"" else s
    }

    companion object {
        private val EXT = setOf("jpg", "jpeg", "png", "bmp", "webp")

        fun listImages(dir: File): List<File> =
            if (!dir.isDirectory) emptyList()
            else dir.walkTopDown().filter { it.isFile && it.extension.lowercase() in EXT }
                .sortedBy { it.relativeTo(dir).path.replace('\\', '/') }.toList()
    }
}
