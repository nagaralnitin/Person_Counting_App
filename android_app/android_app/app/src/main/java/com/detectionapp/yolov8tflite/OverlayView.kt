package com.detectionapp.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var boundingBoxes: List<BoundingBox>? = null
    private val paint = Paint()

    init {
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 8.0f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        boundingBoxes?.forEach { box ->
            paint.color = Color.RED
            paint.textSize = 50.0f
            canvas.drawText("${box.clsName} ${(box.cnf * 100).toInt()}%", box.x1 * width, box.y1 * height - 10, paint)
            canvas.drawRect(box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height, paint)
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        this.boundingBoxes = boundingBoxes
    }
}

