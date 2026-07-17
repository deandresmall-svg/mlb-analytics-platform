from datetime import date,timedelta
import streamlit as st
from mlb_analytics.config import settings
from mlb_analytics.services.pipeline import AnalyticsService
st.title('Historical Backfill');svc=AnalyticsService(settings)
a,b=st.columns(2);start=a.date_input('Start',date.today()-timedelta(days=30));end=b.date_input('End',date.today()-timedelta(days=1))
st.caption('For large backfills, use scripts/backfill.py or GitHub Actions. Streamlit sessions may time out.')
if st.button('Run backfill',type='primary'):
 with st.spinner('Downloading schedules, box scores, weather, and game logs...'):
  try:st.success(svc.backfill(start,end))
  except Exception as e:st.error(str(e))
st.dataframe(svc.repo.coverage(),use_container_width=True,hide_index=True)
