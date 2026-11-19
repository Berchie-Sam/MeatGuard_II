package com.meatguard.app

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Expected values were produced with Pillow 11.3.0:
 *   Image.fromarray(img).resize((3, 3), Image.BILINEAR)   for a random 7 x 5 RGB image.
 * Run with: gradlew.bat testDebugUnitTest
 */
class PilResizeTest {

    private val inRgb = intArrayOf(
        63,13,94,139,133,52,145,240,50,219,137,30,201,85,130,145,188,159,248,178,199,119,255,102,
        104,229,80,210,24,150,60,83,173,182,220,255,225,170,211,179,139,196,1,226,42,238,11,215,
        73,120,245,251,234,99,72,55,165,42,49,214,187,134,225,50,89,183,54,52,36,253,144,7,4,126,
        198,190,159,241,179,43,67,71,232,61,194,227,96,231,199,169,138,178,244,156,168,64,223,157,
        195,72,235,152,14,177,117
    )
    private val outRgb = intArrayOf(
        121,148,82,166,136,117,196,165,188,116,124,130,142,133,163,132,113,185,154,154,126,147,155,
        148,109,171,129
    )

    private fun pack(rgb: IntArray) = IntArray(rgb.size / 3) {
        (0xFF shl 24) or (rgb[3 * it] shl 16) or (rgb[3 * it + 1] shl 8) or rgb[3 * it + 2]
    }

    @Test
    fun matchesPillowBilinear() {
        val out = PilResize.resize(pack(inRgb), 7, 5, 3, 3)
        assertArrayEquals(pack(outRgb), out)
    }

    @Test
    fun sameSizeIsIdentity() {
        val px = pack(inRgb)
        assertArrayEquals(px, PilResize.resize(px, 7, 5, 7, 5))
    }

    @Test
    fun softmaxOfLogits() {
        val p = Prediction.fromLogits(0f, 0f, 0)
        assertEquals(0.5f, p.pSpoiled, 1e-6f)
        val q = Prediction.fromLogits(-1f, 2f, 0)
        assertEquals(0.952574f, q.pSpoiled, 1e-5f)
        assertEquals(true, q.isSpoiled)
    }
}
