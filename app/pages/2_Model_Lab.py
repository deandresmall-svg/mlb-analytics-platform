import streamlit as st
from mlb_analytics.config import settings
from mlb_analytics.services.pipeline import AnalyticsService
st.title('Model Lab');svc=AnalyticsService(settings)
if st.button('Train home win, hit, HR, and strikeout models',type='primary'):st.json(svc.train_all())
g,b,k=svc.datasets();a,c,d=st.columns(3);a.metric('Game rows',len(g));c.metric('Batter-game rows',len(b));d.metric('Starter-game rows',len(k))
st.subheader('Game feature sample');st.dataframe(g.tail(100),use_container_width=True,hide_index=True)
