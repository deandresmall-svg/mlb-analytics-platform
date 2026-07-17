import streamlit as st
from mlb_analytics.config import settings
from mlb_analytics.services.pipeline import AnalyticsService
st.title('Data Quality');svc=AnalyticsService(settings);st.dataframe(svc.repo.coverage(),use_container_width=True,hide_index=True)
for table in ['games','team_game_stats','pitcher_game_stats','player_game_batting']:
 st.subheader(table);st.dataframe(svc.repo.query(f'SELECT * FROM {table} ORDER BY game_date DESC LIMIT 100'),use_container_width=True,hide_index=True)
