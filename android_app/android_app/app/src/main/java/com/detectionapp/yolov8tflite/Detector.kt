package com.detectionapp.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    public var tensorWidth = 0
    public var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        options.numThreads = 4
        interpreter = Interpreter(model, options)

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter?.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        // Count persons
        val personCount = bestBoxes.count { it.clsName == "person" }

        // Filter out non-person boxes
        val personBoxes = bestBoxes.filter { it.clsName == "person" }

        detectorListener.onDetect(personBoxes, inferenceTime, personCount)
    }

    private fun bestBox(array: FloatArray): List<BoundingBox>? {

        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel) {
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]

                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val selected = mutableListOf<BoundingBox>()
        val active = BooleanArray(boxes.size)
        boxes.indices.forEach { active[it] = true }

        boxes.forEachIndexed { i, box ->
            if (active[i]) {
                selected.add(box)
                (i + 1 until boxes.size).forEach { j ->
                    if (active[j]) {
                        val iou = calculateIoU(box, boxes[j])
                        if (iou > IOU_THRESHOLD) active[j] = false
                    }
                }
            }
        }

        return selected
    }

    private fun calculateIoU(a: BoundingBox, b: BoundingBox): Float {
        val aXMin = a.x1
        val aYMin = a.y1
        val aXMax = a.x2
        val aYMax = a.y2
        val bXMin = b.x1
        val bYMin = b.y1
        val bXMax = b.x2
        val bYMax = b.y2

        val areaA = (aXMax - aXMin) * (aYMax - aYMin)
        val areaB = (bXMax - bXMin) * (bYMax - bYMin)

        if (areaA <= 0 || areaB <= 0) return 0.0F

        val intersectionMinX = maxOf(aXMin, bXMin)
        val intersectionMinY = maxOf(aYMin, bYMin)
        val intersectionMaxX = minOf(aXMax, bXMax)
        val intersectionMaxY = minOf(aYMax, bYMax)

        val intersectionArea = maxOf(intersectionMaxX - intersectionMinX, 0F) * maxOf(intersectionMaxY - intersectionMinY, 0F)

        return intersectionArea / (areaA + areaB - intersectionArea)
    }

    interface DetectorListener {
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long, personCount: Int)
        fun onEmptyDetect()
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.5F
        private const val IOU_THRESHOLD = 0.45F
        private const val INPUT_MEAN = 0F
        private const val INPUT_STANDARD_DEVIATION = 255F
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
    }
}
