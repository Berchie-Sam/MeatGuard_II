package com.meatguard.app

/**
 * Steps 3-4 of the image pipeline from a given meat mask (Requirements, Section 3), ported
 * line by line from src/data/segment_meat.py (crop_window, crop_to_mask):
 *   Step 3  every pixel outside the mask is set to the fill colour RGB(124, 116, 104);
 *   Step 4  square window around the mask's bounding box, enlarged by 8% per side and shifted
 *           back into the frame; parts outside the frame are fill as well.
 * Step 5 (bilinear resize of the square to 224) is PilResize.
 *
 * Used now by batch evaluation mode (b) with masks supplied by the research side; the on-device
 * segmentation model (future work) will feed the same functions.
 * Pure Kotlin (no Android classes) so it can be unit-tested on the JVM.
 */
object MaskCrop {
    const val FILL_R = 124
    const val FILL_G = 116
    const val FILL_B = 104
    const val FILL_ARGB = (0xFF shl 24) or (FILL_R shl 16) or (FILL_G shl 8) or FILL_B
    const val PAD = 0.08

    /** Masks covering less than this share of the frame give "No meat detected" (Section 3, Step 2). */
    const val SMALL_MASK = 0.02

    /** Square window (left, top, side) in frame coordinates. */
    data class Window(val left: Int, val top: Int, val side: Int)

    /** Share of the frame covered by the mask (Python: mask.mean()). */
    fun area(mask: BooleanArray): Double = mask.count { it }.toDouble() / mask.size

    /** Python crop_window(mask, pad). Returns null for an empty mask (Python would fail). */
    fun window(mask: BooleanArray, w: Int, h: Int, pad: Double = PAD): Window? {
        require(mask.size == w * h)
        var x0 = Int.MAX_VALUE; var y0 = Int.MAX_VALUE; var x1 = -1; var y1 = -1
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                if (mask[row + x]) {
                    if (x < x0) x0 = x
                    if (x > x1) x1 = x
                    if (y < y0) y0 = y
                    if (y > y1) y1 = y
                }
            }
        }
        if (x1 < 0) return null
        x1 += 1; y1 += 1                                    // exclusive, as ys.max() + 1
        // side = min(int(max(y1 - y0, x1 - x0) * (1 + 2 * pad)), max(h, w)); int() truncates
        val side = minOf((maxOf(y1 - y0, x1 - x0) * (1 + 2 * pad)).toInt(), maxOf(h, w))
        val cy = (y0 + y1) / 2                              // non-negative, so / equals //
        val cx = (x0 + x1) / 2
        val left = (cx - side / 2).coerceIn(0, maxOf(w - side, 0))
        val top = (cy - side / 2).coerceIn(0, maxOf(h - side, 0))
        return Window(left, top, side)
    }

    /**
     * Python: masked[~mask] = BG_FILL; canvas = Image.new("RGB", (side, side), BG_FILL);
     * canvas.paste(masked, (-left, -top)). Returns the side x side ARGB canvas.
     */
    fun fillAndCrop(px: IntArray, mask: BooleanArray, w: Int, h: Int, win: Window): IntArray {
        require(px.size == w * h && mask.size == w * h)
        val s = win.side
        val out = IntArray(s * s)
        for (yy in 0 until s) {
            val y = yy + win.top
            val o = yy * s
            if (y >= h) {
                java.util.Arrays.fill(out, o, o + s, FILL_ARGB)
                continue
            }
            val row = y * w
            for (xx in 0 until s) {
                val x = xx + win.left
                out[o + xx] = if (x < w && mask[row + x]) (px[row + x] or (0xFF shl 24)) else FILL_ARGB
            }
        }
        return out
    }
}
