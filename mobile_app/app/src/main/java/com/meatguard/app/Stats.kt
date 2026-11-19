package com.meatguard.app

import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Summary of a set of timings (Requirements 6.6). Percentiles use numpy's default definition
 * (linear interpolation between order statistics), so they equal np.percentile on the CSV.
 * Pure Kotlin so it can be unit-tested on the JVM.
 */
data class Summary(
    val n: Int, val mean: Double, val sd: Double, val min: Double, val q1: Double,
    val median: Double, val q3: Double, val p90: Double, val max: Double
) {
    val iqr: Double get() = q3 - q1

    fun line(unit: String = "ms") =
        "median %.2f %s (IQR %.2f-%.2f), mean %.2f ± %.2f, p90 %.2f, min %.2f, max %.2f, n=%d"
            .format(median, unit, q1, q3, mean, sd, p90, min, max, n)

    companion object {
        val HEADER = listOf("n", "mean", "sd", "min", "q1", "median", "q3", "iqr", "p90", "max")

        fun of(values: List<Double>): Summary {
            require(values.isNotEmpty())
            val v = values.sorted()
            val n = v.size
            val mean = v.sum() / n
            // Sample standard deviation (ddof = 1), as pandas .std()
            val sd = if (n > 1) sqrt(v.sumOf { (it - mean) * (it - mean) } / (n - 1)) else 0.0
            return Summary(n, mean, sd, v.first(), pct(v, 25.0), pct(v, 50.0), pct(v, 75.0), pct(v, 90.0), v.last())
        }

        /** np.percentile(v, q) with the default "linear" method; [v] sorted. */
        fun pct(v: List<Double>, q: Double): Double {
            val pos = (v.size - 1) * q / 100.0
            val lo = floor(pos).toInt()
            val hi = minOf(lo + 1, v.size - 1)
            return v[lo] + (v[hi] - v[lo]) * (pos - lo)
        }
    }

    fun values(): List<Any> = listOf(n, mean, sd, min, q1, median, q3, iqr, p90, max)
}
