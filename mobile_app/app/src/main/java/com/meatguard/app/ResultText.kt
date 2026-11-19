package com.meatguard.app

import java.util.Locale

/**
 * Fixed, offline texts. They describe only what the model can judge: the visual appearance of
 * the meat surface. The training labels came from visual inspection, not from laboratory tests,
 * so no claim about bacteria, odour or safety is made.
 */
object ResultText {

    const val DISCLAIMER =
        "MeatGuard is a visual screening aid, not a laboratory test. It judges only how the meat " +
        "looks in the photo, not its smell, temperature or storage history. If in doubt, do not eat the meat."

    const val FRAMING_TIP =
        "Take or choose a photo, then draw the square tightly around the meat with a little space " +
        "around it. Avoid trays, bags and other objects in the square."

    fun title(label: String, confidence: Float): String =
        "${if (label == "spoiled") "SPOILED" else "FRESH"}  (${percent(confidence)})"

    /** Probability of the predicted class (0.5..1). Never shown as 100%: a model is not certain. */
    fun percent(p: Float): String =
        if (p >= 0.995f) "> 99%" else String.format(Locale.US, "%.0f%%", p * 100f)

    fun explanation(isSpoiled: Boolean, isUncertain: Boolean): String {
        val base = if (isSpoiled)
            "The meat in the square looks like the spoiled meat the model was trained on " +
            "(changes in colour, texture or surface condition)."
        else
            "The meat in the square looks like the fresh meat the model was trained on."
        val unsure = if (isUncertain)
            "\n\nThe model is uncertain about this photo (probability between 40% and 60%). " +
            "Retake the photo in good light, framing only the meat, or inspect the meat carefully."
        else ""
        return base + unsure
    }

    fun inferenceTime(micros: Long): String =
        String.format(Locale.US, "Model time: %.1f ms", micros / 1000.0)
}
