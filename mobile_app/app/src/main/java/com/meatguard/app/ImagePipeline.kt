package com.meatguard.app

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream

/**
 * Image steps of the app (Requirements, Section 3):
 *   Step 1  decode + EXIF orientation          -> decodeUpright()
 *   Step 2  (interim) user's square crop       -> done by the crop screen before this
 *   Step 5  bilinear resize to 224 (Pillow)    -> toModelPixels()
 *   Step 6  normalisation                       -> Classifier.setInput()
 * Steps 3-4 (fill + square crop from a given mask) are in MaskCrop; the app uses them in batch
 * evaluation mode (b) until the segmentation model (future work) supplies the mask.
 */
object ImagePipeline {

    /** Decode an image as ARGB_8888 and rotate/flip it upright according to its EXIF tag. */
    fun decodeUpright(resolver: ContentResolver, uri: Uri): Bitmap {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bmp = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
            ?: throw IllegalArgumentException("Cannot read the image")
        val orientation = try {
            resolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL   // e.g. PNG without EXIF
        }
        return orient(bmp, orientation)
    }

    /**
     * Decode a file as ARGB_8888. With [applyExif] = false the stored pixel order is kept, as
     * PIL's Image.open(...).convert("RGB") did when the research masks and crops were made.
     */
    fun decodeFile(file: File, applyExif: Boolean): Bitmap {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bmp = BitmapFactory.decodeFile(file.absolutePath, opts)
            ?: throw IllegalArgumentException("Cannot read ${file.name}")
        if (!applyExif) return bmp
        val orientation = try {
            ExifInterface(file.absolutePath).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        return orient(bmp, orientation)
    }

    private fun orient(bmp: Bitmap, orientation: Int): Bitmap {
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
            else -> return bmp
        }
        val upright = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, false)
        if (upright !== bmp) bmp.recycle()
        return upright
    }

    /**
     * Square crop -> 224 x 224 ARGB pixels, resized exactly as Pillow did for training.
     * If the crop is not exactly square (a pixel of rounding), the centre square is used.
     */
    fun toModelPixels(crop: Bitmap): IntArray {
        val side = minOf(crop.width, crop.height)
        val left = (crop.width - side) / 2
        val top = (crop.height - side) / 2
        val px = IntArray(side * side)
        crop.getPixels(px, 0, side, left, top, side, side)
        return PilResize.resize(px, side, side, ModelInfo.INPUT_SIZE, ModelInfo.INPUT_SIZE)
    }

    fun pixelsToBitmap(px: IntArray, w: Int, h: Int): Bitmap =
        Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)

    /** Copy for display in the history, at most [maxSide] pixels (display only, not analysed). */
    fun displayCopy(src: Bitmap, maxSide: Int = 512): Bitmap {
        val s = maxOf(src.width, src.height)
        if (s <= maxSide) return src
        val f = maxSide.toFloat() / s
        return Bitmap.createScaledBitmap(src, (src.width * f).toInt(), (src.height * f).toInt(), true)
    }

    fun save(bitmap: Bitmap, file: File, format: Bitmap.CompressFormat, quality: Int = 90) {
        file.parentFile?.mkdirs()
        FileOutputStream(file).use { bitmap.compress(format, quality, it) }
    }
}
