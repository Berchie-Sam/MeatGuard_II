package com.meatguard.app

import android.content.Context
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.math.exp
import kotlin.math.max

/** Facts about the bundled model, shown in About and stored with every report. */
object ModelInfo {
    const val FILE = "meatguard_s_int8_float_attn.tflite"
    const val NAME = "MeatGuard-S (pruned, 8-bit)"
    const val VERSION = "specimen split, v1"
    const val EXPECTED_SHA256 = "7f1ae042c3680ba551ae3341e4729b1ae81f867adbe8ba8592e497a1e6a47090"
    const val INPUT_SIZE = 224

    /** Interim pipeline: user's square crop, no segmentation (Requirements, Section 3, Step 2). */
    const val PIPELINE = "manual-crop-v1"

    /** ImageNet normalisation used in training: (pixel / 255 - mean) / std, R, G, B. */
    val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** Probabilities in this band are shown as "uncertain"; the label still uses 0.5. */
    const val UNCERTAIN_LOW = 0.4f
    const val UNCERTAIN_HIGH = 0.6f

    fun sha256(context: Context): String {
        val md = MessageDigest.getInstance("SHA-256")
        context.assets.open(FILE).use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}

/** Output of one inference. Class order of the model: index 0 = fresh, 1 = spoiled. */
data class Prediction(
    val logitFresh: Float,
    val logitSpoiled: Float,
    val pSpoiled: Float,
    val inferenceMicros: Long
) {
    val pFresh: Float get() = 1f - pSpoiled
    val isSpoiled: Boolean get() = pSpoiled >= 0.5f
    val isUncertain: Boolean get() = pSpoiled >= ModelInfo.UNCERTAIN_LOW && pSpoiled <= ModelInfo.UNCERTAIN_HIGH
    /** Probability of the predicted class, 0.5..1 */
    val confidence: Float get() = if (isSpoiled) pSpoiled else pFresh

    companion object {
        /** The model outputs logits; softmax over the two classes (numerically stable). */
        fun fromLogits(zFresh: Float, zSpoiled: Float, micros: Long): Prediction {
            val m = max(zFresh, zSpoiled)
            val eF = exp((zFresh - m).toDouble())
            val eS = exp((zSpoiled - m).toDouble())
            return Prediction(zFresh, zSpoiled, (eS / (eF + eS)).toFloat(), micros)
        }
    }
}

/** Execution settings of Requirements 6.2. The app itself uses CPU4. */
enum class Accel(val label: String) {
    CPU1("CPU, 1 thread (XNNPACK)"),
    CPU4("CPU, 4 threads (XNNPACK)"),
    NNAPI("NNAPI delegate"),
    GPU("GPU delegate")
}

/**
 * Runs the deployed TensorFlow Lite model on a 224 x 224 crop.
 * Not thread-safe: call from one background thread at a time.
 *
 * If a delegate cannot be created or applied, the constructor throws (the benchmark reports it);
 * it never falls back to the CPU silently. Partial delegation (some operations left on the CPU)
 * is not visible through the Java API; [notes] says so for the delegate settings.
 */
class Classifier(context: Context, val accel: Accel = Accel.CPU4) : Closeable {

    val modelSha256: String = ModelInfo.sha256(context)
    /** Interpreter creation time including the delegate, in microseconds. */
    val loadMicros: Long
    val notes: String
    private val interpreter: Interpreter
    private val delegate: Delegate?
    private val input: ByteBuffer
    private val output = Array(1) { FloatArray(2) }

    init {
        val options = Interpreter.Options()
        var note = ""
        delegate = when (accel) {
            Accel.CPU1 -> { options.setNumThreads(1); options.setUseXNNPACK(true); null }
            Accel.CPU4 -> { options.setNumThreads(4); options.setUseXNNPACK(true); null }
            Accel.NNAPI -> {
                note = "partial delegation not visible from the app"
                NnApiDelegate(NnApiDelegate.Options()
                    .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER))
            }
            Accel.GPU -> {
                val supported = try { CompatibilityList().use { it.isDelegateSupportedOnThisDevice } } catch (e: Throwable) { false }
                note = "GPU compatibility list: ${if (supported) "supported" else "not listed"}; " +
                       "partial delegation not visible from the app"
                GpuDelegate()
            }
        }
        if (delegate != null) options.addDelegate(delegate)
        notes = note
        val model = loadModel(context)
        val t0 = System.nanoTime()
        interpreter = try {
            Interpreter(model, options)
        } catch (e: Throwable) {
            delegate?.close()
            throw IllegalStateException("${accel.label}: could not be created or applied (${e.message})", e)
        }
        loadMicros = (System.nanoTime() - t0) / 1000
        val inShape = interpreter.getInputTensor(0).shape()
        val outShape = interpreter.getOutputTensor(0).shape()
        check(inShape.contentEquals(intArrayOf(1, ModelInfo.INPUT_SIZE, ModelInfo.INPUT_SIZE, 3))) {
            "Unexpected model input shape ${inShape.contentToString()}"
        }
        check(outShape.contentEquals(intArrayOf(1, 2))) {
            "Unexpected model output shape ${outShape.contentToString()}"
        }
        input = ByteBuffer.allocateDirect(4 * ModelInfo.INPUT_SIZE * ModelInfo.INPUT_SIZE * 3)
            .order(ByteOrder.nativeOrder())
    }

    private fun loadModel(context: Context): MappedByteBuffer {
        context.assets.openFd(ModelInfo.FILE).use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /** [pixels]: 224 x 224 ARGB pixels, row-major (as from PilResize.resize). */
    fun classify(pixels: IntArray): Prediction {
        setInput(pixels)
        val ns = runOnce()
        return Prediction.fromLogits(output[0][0], output[0][1], ns / 1000)
    }

    /** Step 6: NHWC float32, value = (pixel / 255 - mean) / std per channel (R, G, B). */
    fun setInput(pixels: IntArray) {
        require(pixels.size == ModelInfo.INPUT_SIZE * ModelInfo.INPUT_SIZE)
        input.rewind()
        val m = ModelInfo.MEAN
        val s = ModelInfo.STD
        for (p in pixels) {
            input.putFloat((((p shr 16) and 0xFF) / 255f - m[0]) / s[0])
            input.putFloat((((p shr 8) and 0xFF) / 255f - m[1]) / s[1])
            input.putFloat(((p and 0xFF) / 255f - m[2]) / s[2])
        }
        input.rewind()
    }

    /** Step 7 only: one interpreter.run() on the current input; returns its duration in ns. */
    fun runOnce(): Long {
        input.rewind()
        val t0 = System.nanoTime()
        interpreter.run(input, output)
        return System.nanoTime() - t0
    }

    /** Duration of the last run as measured inside the runtime (ns), or -1 if not available. */
    fun lastNativeNanos(): Long = try { interpreter.lastNativeInferenceDurationNanoseconds ?: -1L } catch (e: Throwable) { -1L }

    /** Logits [fresh, spoiled] of the last run. */
    fun logits(): FloatArray = floatArrayOf(output[0][0], output[0][1])

    /** NNAPI reports errors after a run; null if none or not applicable. */
    fun delegateErrors(): String? {
        val d = delegate as? NnApiDelegate ?: return null
        return if (d.hasErrors()) "NNAPI error ${d.nnapiErrno}" else null
    }

    override fun close() {
        interpreter.close()
        delegate?.close()
    }
}
