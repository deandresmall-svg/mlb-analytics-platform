from __future__ import annotations

import streamlit as st

from mlb_analytics.config import settings
from mlb_analytics.evaluation.cheatsheet import load_reliability_snapshot
from mlb_analytics.ui.cheatsheet import render_cheat_sheet


st.set_page_config(page_title="Live Model Cheat Sheet", page_icon="📘", layout="wide")
st.title("📘 Live Model Cheat Sheet")
st.caption(
    "This page reads the latest chronological holdout reliability snapshot. "
    "Every completed Calibration evaluation rewrites the cheat sheet, so the "
    "thresholds, observed rates, samples, and warnings follow the currently saved models."
)

snapshot = load_reliability_snapshot(settings.reliability_snapshot_path)
render_cheat_sheet(snapshot, settings.model_dir, compact=False)

st.info(
    "After training new models, open Calibration and run the holdout reliability "
    "evaluation once. Until then, the dashboard will mark an older cheat sheet as stale."
)
