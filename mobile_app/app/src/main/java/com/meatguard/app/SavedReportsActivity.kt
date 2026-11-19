package com.meatguard.app

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.meatguard.app.databinding.ActivitySavedReportsBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SavedReportsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySavedReportsBinding
    private val adapter = ReportsAdapter { showReport(it) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ReportStorage.init(this)
        binding = ActivitySavedReportsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Saved Reports"
        binding.toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }

        binding.recyclerViewReports.layoutManager = LinearLayoutManager(this)
        binding.recyclerViewReports.adapter = adapter
        refresh()
    }

    private fun refresh() = adapter.submit(ReportStorage.all())

    private fun showReport(r: Report) {
        val v = LayoutInflater.from(this).inflate(R.layout.dialog_report_detail, null)
        val dialog = AlertDialog.Builder(this).setView(v).create()

        v.findViewById<TextView>(R.id.detailTitle).apply {
            text = ResultText.title(r.label, r.confidence)
            setTextColor(ContextCompat.getColor(context, if (r.isSpoiled) R.color.result_spoiled else R.color.result_fresh))
        }
        v.findViewById<TextView>(R.id.detailDateTime).text = formatDate(r.timestamp, "MMMM dd, yyyy 'at' hh:mm a")
        r.displayImagePath?.let { path ->
            BitmapFactory.decodeFile(path)?.let { v.findViewById<ImageView>(R.id.detailImageView).setImageBitmap(it) }
        }
        val uncertain = r.pSpoiled >= ModelInfo.UNCERTAIN_LOW && r.pSpoiled <= ModelInfo.UNCERTAIN_HIGH
        v.findViewById<TextView>(R.id.detailExplanation).text =
            ResultText.explanation(r.isSpoiled, uncertain) + "\n\n" + ResultText.DISCLAIMER
        v.findViewById<TextView>(R.id.detailInfo).text =
            "P(spoiled) = ${"%.3f".format(r.pSpoiled)}; ${ResultText.inferenceTime(r.inferenceMicros)}\n" +
            "Model SHA-256: ${r.modelSha256.take(16)}...\nPipeline: ${r.pipeline}; app ${r.appVersion}"

        v.findViewById<Button>(R.id.detailDeleteButton).setOnClickListener {
            AlertDialog.Builder(this)
                .setMessage("Delete this report?")
                .setPositiveButton("Delete") { _, _ ->
                    ReportStorage.delete(this, r.id)
                    refresh()
                    dialog.dismiss()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
        v.findViewById<Button>(R.id.detailCloseButton).setOnClickListener { dialog.dismiss() }
        dialog.show()
    }
}

fun formatDate(ts: Long, pattern: String): String = SimpleDateFormat(pattern, Locale.getDefault()).format(Date(ts))

class ReportsAdapter(private val onClick: (Report) -> Unit) : RecyclerView.Adapter<ReportsAdapter.VH>() {

    private var items: List<Report> = emptyList()

    class VH(view: android.view.View) : RecyclerView.ViewHolder(view) {
        val thumb: ImageView = view.findViewById(R.id.item_thumbnail)
        val result: TextView = view.findViewById(R.id.item_result)
        val date: TextView = view.findViewById(R.id.item_date)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(LayoutInflater.from(parent.context).inflate(R.layout.item_report, parent, false))

    override fun onBindViewHolder(holder: VH, position: Int) {
        val r = items[position]
        holder.result.text = ResultText.title(r.label, r.confidence)
        holder.result.setTextColor(
            ContextCompat.getColor(holder.itemView.context, if (r.isSpoiled) R.color.result_spoiled else R.color.result_fresh)
        )
        holder.date.text = formatDate(r.timestamp, "MMM dd, yyyy\nhh:mm a")
        // The 224 x 224 model input is small enough to decode directly for the thumbnail.
        val thumb = r.modelImagePath?.let { BitmapFactory.decodeFile(it) }
        holder.thumb.setImageBitmap(thumb)
        holder.itemView.setOnClickListener { onClick(r) }
    }

    override fun getItemCount() = items.size

    @android.annotation.SuppressLint("NotifyDataSetChanged")
    fun submit(list: List<Report>) {
        items = list
        notifyDataSetChanged()
    }
}
