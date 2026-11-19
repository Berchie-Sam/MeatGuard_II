package com.meatguard.app

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max

/**
 * Bilinear resize identical to Pillow's Image.resize(size, Image.BILINEAR) for 8-bit RGB
 * (Pillow src/libImaging/Resample.c): separable, horizontal pass first, 22-bit fixed-point
 * weights, rounding to 8 bits between the passes. When shrinking, the filter widens with the
 * scale factor (antialiasing), unlike Bitmap.createScaledBitmap.
 *
 * The training crops were made with Pillow, so this keeps the app input identical to training.
 * Pure Kotlin (no Android classes) so it can be unit-tested on the JVM.
 * Pixels are ARGB ints (as from Bitmap.getPixels); alpha is ignored and set to 255.
 */
object PilResize {
    private const val PRECISION_BITS = 32 - 8 - 2

    private class Coeffs(val bounds: IntArray, val kk: IntArray, val ksize: Int)

    private fun precompute(inSize: Int, outSize: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterScale = max(scale, 1.0)
        val support = 1.0 * filterScale              // bilinear filter support = 1
        val ksize = ceil(support).toInt() * 2 + 1
        val ss = 1.0 / filterScale
        val bounds = IntArray(outSize * 2)
        val kk = IntArray(outSize * ksize)
        val k = DoubleArray(ksize)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            var xmin = (center - support + 0.5).toInt()   // toInt truncates toward zero, as C
            if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt()
            if (xmax > inSize) xmax = inSize
            xmax -= xmin
            var ww = 0.0
            for (x in 0 until ksize) k[x] = 0.0
            for (x in 0 until xmax) {
                val t = abs((x + xmin - center + 0.5) * ss)
                val w = if (t < 1.0) 1.0 - t else 0.0
                k[x] = w
                ww += w
            }
            for (x in 0 until xmax) {
                val w = if (ww != 0.0) k[x] / ww else 0.0
                kk[xx * ksize + x] = if (w < 0) (-0.5 + w * (1 shl PRECISION_BITS)).toInt()
                                     else (0.5 + w * (1 shl PRECISION_BITS)).toInt()
            }
            bounds[xx * 2] = xmin
            bounds[xx * 2 + 1] = xmax
        }
        return Coeffs(bounds, kk, ksize)
    }

    private fun clip8(v: Int): Int {
        val s = v shr PRECISION_BITS
        return if (s < 0) 0 else if (s > 255) 255 else s
    }

    /** Resize [src] (width [w], height [h]) to [outW] x [outH]. Returns ARGB pixels. */
    fun resize(src: IntArray, w: Int, h: Int, outW: Int, outH: Int): IntArray {
        require(src.size == w * h) { "pixel array size does not match $w x $h" }
        var cw = w
        var ch = h
        val half = 1 shl (PRECISION_BITS - 1)
        val r: IntArray; val g: IntArray; val b: IntArray

        if (outW != cw) {
            // Horizontal pass reads the channels straight from the ARGB source, so a large photo
            // is not copied into three full-size planes first (memory; same arithmetic).
            val c = precompute(cw, outW)
            r = IntArray(outW * ch); g = IntArray(outW * ch); b = IntArray(outW * ch)
            for (y in 0 until ch) {
                val row = y * cw
                for (xx in 0 until outW) {
                    val xmin = c.bounds[xx * 2]
                    val xmax = c.bounds[xx * 2 + 1]
                    val ko = xx * c.ksize
                    var sr = half; var sg = half; var sb = half
                    for (x in 0 until xmax) {
                        val kv = c.kk[ko + x]
                        val p = src[row + x + xmin]
                        sr += ((p shr 16) and 0xFF) * kv; sg += ((p shr 8) and 0xFF) * kv; sb += (p and 0xFF) * kv
                    }
                    val o = y * outW + xx
                    r[o] = clip8(sr); g[o] = clip8(sg); b[o] = clip8(sb)
                }
            }
            cw = outW
        } else {
            r = IntArray(w * h) { (src[it] shr 16) and 0xFF }
            g = IntArray(w * h) { (src[it] shr 8) and 0xFF }
            b = IntArray(w * h) { src[it] and 0xFF }
        }
        var rr = r; var gg = g; var bb = b
        if (outH != ch) {
            val c = precompute(ch, outH)
            val nr = IntArray(cw * outH); val ng = IntArray(cw * outH); val nb = IntArray(cw * outH)
            for (yy in 0 until outH) {
                val ymin = c.bounds[yy * 2]
                val ymax = c.bounds[yy * 2 + 1]
                val ko = yy * c.ksize
                for (x in 0 until cw) {
                    var sr = half; var sg = half; var sb = half
                    for (y in 0 until ymax) {
                        val kv = c.kk[ko + y]
                        val i = (y + ymin) * cw + x
                        sr += rr[i] * kv; sg += gg[i] * kv; sb += bb[i] * kv
                    }
                    val o = yy * cw + x
                    nr[o] = clip8(sr); ng[o] = clip8(sg); nb[o] = clip8(sb)
                }
            }
            rr = nr; gg = ng; bb = nb; ch = outH
        }
        return IntArray(cw * ch) { (0xFF shl 24) or (rr[it] shl 16) or (gg[it] shl 8) or bb[it] }
    }
}
