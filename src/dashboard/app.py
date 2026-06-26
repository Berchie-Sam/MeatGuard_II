import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

# This app is designed to run from the root of MeatGuard_II
# Command: streamlit run src/dashboard/app.py

st.set_page_config(page_title="MeatGuard-S Research Dashboard", layout="wide")

st.title("🥩 MeatGuard-S Research Dashboard")
st.sidebar.header("Controls")

# Load config to get paths
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    output_path = config['visualization']['output_path']

metrics_file = Path(output_path) / "dashboard" / "metrics_log.jsonl"

if metrics_file.exists():
    data = []
    with open(metrics_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    
    st.subheader("Real-time Pipeline Metrics")
    # Plotting logic based on metric_name
    fig = px.line(df, x=df.index, y="value", color="metric", title="Metric Evolution")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No metrics logged yet. Run a pipeline stage to see data.")

st.sidebar.markdown("---")
st.sidebar.write(f"**Current Mode:** {config['mode']}")
st.sidebar.write(f"**Architecture:** {config['architecture']}")
