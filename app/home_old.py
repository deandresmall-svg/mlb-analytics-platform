from datetime import date
import streamlit as st
from mlb_analytics.config import settings
from mlb_analytics.services.pipeline import AnalyticsService
st.set_page_config(page_title='MLB Analytics',page_icon='⚾',layout='wide')
st.title('⚾ MLB Analytics Dashboard');st.caption('GitHub-ready Streamlit app with point-in-time features and separate game/player models')
svc=AnalyticsService(settings);d=st.date_input('Slate date',date.today())
a,b,c=st.columns(3)
if a.button('Sync slate',type='primary',use_container_width=True):
 try:svc.sync_schedule(d);st.success('Slate updated')
 except Exception as e:st.error(str(e))
if b.button('Train all models',use_container_width=True):st.json(svc.train_all())
c.metric('Database games',int(svc.repo.coverage().games.iloc[0]))
g=svc.game_predictions(d)
if g.empty:st.info('No games stored for this date. Sync the slate first.')
else:
 cols=['game_time','away_team','home_team','away_probable_pitcher','home_probable_pitcher','venue','status']
 if 'home_win_probability' in g:
  cols+=['pick','confidence'];g['confidence']=g.confidence.map(lambda x:f'{x:.1%}')
 st.dataframe(g[cols],use_container_width=True,hide_index=True)
st.warning('Models require a historical backfill before their probabilities are meaningful. No slip optimizer is included.')
