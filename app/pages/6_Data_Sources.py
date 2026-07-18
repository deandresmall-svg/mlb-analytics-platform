from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from mlb_analytics.config import settings
from mlb_analytics.services.pipeline import AnalyticsService


st.set_page_config(page_title="Data Sources", page_icon="🛰️", layout="wide")
st.title("🛰️ Data Sources")
st.caption(
    "Sync and validate Baseball Savant Statcast data. Downloads are stored "
    "pitch by pitch, then aggregated by player, game, pitch type, and handedness."
)

service = AnalyticsService(settings)
coverage = service.repo.coverage()
coverage_row = coverage.iloc[0] if not coverage.empty else pd.Series(dtype=object)

summary = st.columns(5)
summary[0].metric("Statcast pitches", f"{int(coverage_row.get('statcast_pitches', 0) or 0):,}")
summary[1].metric(
    "Batter-game rows",
    f"{int(coverage_row.get('statcast_batter_games', 0) or 0):,}",
)
summary[2].metric(
    "Pitcher-game rows",
    f"{int(coverage_row.get('statcast_pitcher_games', 0) or 0):,}",
)
summary[3].metric("First date", coverage_row.get("statcast_start") or "None")
summary[4].metric("Last date", coverage_row.get("statcast_end") or "None")

st.divider()
st.subheader("Sync Statcast")
st.info(
    "Before retraining, sync the same historical range used by your boxscore "
    "backfill. For the current database, that usually means starting at the "
    "first backfilled game rather than syncing only the latest few weeks."
)

default_end = date.today() - timedelta(days=1)
default_start = default_end - timedelta(
    days=max(settings.statcast_default_sync_days - 1, 0)
)

controls = st.columns([1, 1, 1])
with controls[0]:
    start_date = st.date_input("Start date", default_start)
with controls[1]:
    end_date = st.date_input("End date", default_end)
with controls[2]:
    chunk_days = st.number_input(
        "Days per download chunk",
        min_value=1,
        max_value=7,
        value=max(1, min(settings.statcast_chunk_days, 7)),
        step=1,
        help="Smaller chunks are slower but less likely to time out.",
    )

sync_col, rebuild_col = st.columns(2)
with sync_col:
    sync_clicked = st.button(
        "Sync Statcast range",
        type="primary",
        width="stretch",
        disabled=end_date < start_date,
    )
with rebuild_col:
    rebuild_clicked = st.button(
        "Rebuild aggregates from stored pitches",
        width="stretch",
        disabled=end_date < start_date,
    )

if sync_clicked:
    bar = st.progress(0.0, text="Preparing Statcast sync...")
    status = st.empty()

    def update_progress(done: int, total: int, message: str) -> None:
        fraction = done / total if total else 0.0
        bar.progress(min(max(fraction, 0.0), 1.0), text=message)
        status.caption(f"Chunk {min(done + 1, total)} of {total}")

    try:
        result = service.sync_statcast(
            start_date,
            end_date,
            int(chunk_days),
            progress=update_progress,
        )
        bar.progress(1.0, text="Statcast sync complete")
        if result.get("errors"):
            st.warning(
                f"Sync completed with {len(result['errors'])} failed chunk(s). "
                "Successful chunks were saved and can be resumed."
            )
            with st.expander("View failed chunks"):
                for message in result["errors"]:
                    st.write(f"• {message}")
        else:
            st.success("Statcast sync completed successfully.")
        metrics = st.columns(5)
        metrics[0].metric("Pitches", f"{result.get('pitches', 0):,}")
        metrics[1].metric("Batter games", f"{result.get('batter_games', 0):,}")
        metrics[2].metric("Pitcher games", f"{result.get('pitcher_games', 0):,}")
        metrics[3].metric(
            "Batter pitch-type rows",
            f"{result.get('batter_pitch_type_rows', 0):,}",
        )
        metrics[4].metric(
            "Pitcher pitch-type rows",
            f"{result.get('pitcher_pitch_type_rows', 0):,}",
        )
        with st.expander("Chunk details"):
            st.dataframe(pd.DataFrame(result.get("chunks", [])), hide_index=True)
    except Exception as exc:
        st.error(f"Statcast sync failed: {exc}")

if rebuild_clicked:
    try:
        with st.spinner("Rebuilding Statcast aggregates..."):
            result = service.rebuild_statcast_aggregates(start_date, end_date)
        st.success("Statcast aggregates rebuilt.")
        st.json(result)
    except Exception as exc:
        st.error(f"Aggregate rebuild failed: {exc}")

st.divider()
st.subheader("Recent Statcast coverage")
try:
    recent = service.repo.query(
        """
        SELECT
            game_date,
            COUNT(*) AS pitches,
            COUNT(DISTINCT batter) AS batters,
            COUNT(DISTINCT pitcher) AS pitchers,
            SUM(
                CASE
                    WHEN UPPER(COALESCE(type, '')) = 'X'
                         AND launch_speed IS NOT NULL
                    THEN 1 ELSE 0
                END
            ) AS batted_balls,
            SUM(
                CASE
                    WHEN UPPER(COALESCE(type, '')) = 'X'
                         AND launch_speed >= 95.0
                    THEN 1 ELSE 0
                END
            ) AS hard_hits,
            SUM(
                CASE
                    WHEN UPPER(COALESCE(type, '')) = 'X'
                         AND launch_speed IS NOT NULL
                         AND launch_speed_angle = 6
                    THEN 1 ELSE 0
                END
            ) AS barrels
        FROM statcast_pitches
        GROUP BY game_date
        ORDER BY game_date DESC
        LIMIT 30
        """
    )
except Exception:
    recent = pd.DataFrame()

if recent.empty:
    st.info("No Statcast rows have been synced yet.")
else:
    st.dataframe(recent, width="stretch", hide_index=True)

st.caption(
    "After completing the historical Statcast sync, return to the Home page "
    "and run Train all models. Existing saved models remain usable until then."
)
