import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yaml
import json

class VisualizationManager:
    """
    Unified Visualization Manager for MeatGuard II.
    Integrates static plots, interactive dashboards, and HTML reports.
    """
    def __init__(self, config_manager):
        self.cm = config_manager
        self.viz_cfg = self.cm.get('visualization')
        self.style = self.cm.get('viz_style', {})
        self.engine = self.viz_cfg.get('engine', 'plotly')
        self.modality = self.viz_cfg.get('modality', 'both')
        self.output_root = Path(self.viz_cfg.get('output_path', 'artifacts/outputs'))
        self.save_dir = self.output_root / "plots"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _apply_style(self, ax=None):
        primary = self.style.get('primary_color', '#003366')
        if ax:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        return primary

    def plot_metric_curve(self, x, y, title="Metric Curve", xlabel="Epoch", ylabel="Value", filename="curve.png"):
        if not self.viz_cfg.get('auto_visualize', True): return
        
        color = self._apply_style()
        if self.engine == 'matplotlib_sns':
            plt.figure(figsize=(8, 5))
            sns.set_style("whitegrid")
            plt.plot(x, y, color=color, linewidth=self.style.get('line_width', 2))
            plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
            plt.savefig(self.save_dir / filename)
            plt.close()
        elif self.engine == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(color=color)))
            fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white")
            fig.write_image(str(self.save_dir / filename))

    def generate_pipeline_report(self, stage_name, metrics_dict):
        """
        Triggers the generation of a high-fidelity HTML report for a specific pipeline stage.
        Saves a JSON summary to be consumed by the Visual-Explainer tool.
        """
        report_dir = self.output_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "stage": stage_name,
            "metrics": metrics_dict,
            "timestamp": torch.datetime.datetime.now().isoformat() if hasattr(torch, 'datetime') else "N/A"
        }
        
        with open(report_dir / f"{stage_name}_summary.json", "w") as f:
            json.dump(report_data, f, indent=4)
            
        print(f"Report data for {stage_name} saved to {report_dir}. Ready for Visual-Explainer synthesis.")

    def update_dashboard_data(self, metric_name, value):
        """
        Pushes a metric to the dashboard cache (JSON/CSV) for Streamlit to render.
        """
        if self.modality == 'image_only': return
        
        dash_dir = self.output_root / "dashboard"
        dash_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple append-only log for the dashboard
        with open(dash_dir / "metrics_log.jsonl", "a") as f:
            f.write(json.dumps({"metric": metric_name, "value": value}) + "\n")
