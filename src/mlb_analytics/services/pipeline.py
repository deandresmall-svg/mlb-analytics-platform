from dataclasses import asdict
from datetime import date,timedelta,datetime
import pandas as pd
from mlb_analytics.config import Settings
from mlb_analytics.data.mlb_api import MLBClient
from mlb_analytics.data.weather import WeatherClient
from mlb_analytics.data.venues import venue_info
from mlb_analytics.data.parsers import parse_schedule,parse_boxscore
from mlb_analytics.data.repository import Repository
from mlb_analytics.features.game_features import FEATURES,build_game_features
from mlb_analytics.features.player_features import HIT_FEATURES,HR_FEATURES,K_FEATURES,build_batter_training,build_pitcher_k_training
from mlb_analytics.models.base import BinaryTimeModel,CountTimeModel
class AnalyticsService:
 def __init__(self,s:Settings):
  self.s=s;s.ensure_directories();self.mlb=MLBClient(s.api_base_url,s.api_timeout_seconds,s.api_max_retries);self.weather=WeatherClient(s.weather_base_url,s.weather_archive_url,s.api_timeout_seconds,s.api_max_retries);self.repo=Repository(s.database_url);self.repo.initialize()
 def sync_schedule(self,start:date,end:date|None=None,with_weather=True):
  g=parse_schedule(self.mlb.schedule(start,end));
  if g.empty:return g
  for c in ['temperature','humidity','pressure','wind_speed','wind_direction','precipitation','park_factor']:g[c]=None
  for i,r in g.iterrows():
   lat,lon,pf=venue_info(r.venue);g.at[i,'park_factor']=pf
   if with_weather and lat is not None:
    try:
     wx=self.weather.nearest(self.weather.hourly(lat,lon,date.fromisoformat(r.game_date)),r.game_time)
     g.at[i,'temperature']=wx.get('temperature_2m');g.at[i,'humidity']=wx.get('relative_humidity_2m');g.at[i,'pressure']=wx.get('surface_pressure');g.at[i,'wind_speed']=wx.get('wind_speed_10m');g.at[i,'wind_direction']=wx.get('wind_direction_10m');g.at[i,'precipitation']=wx.get('precipitation')
    except Exception:pass
  self.repo.upsert_games(g);return g
 def backfill(self,start:date,end:date,include_boxscores=True):
  synced=0; boxes=0; day=start
  while day<=end:
   games=self.sync_schedule(day,day,with_weather=True);synced+=len(games)
   if include_boxscores:
    for _,gr in games.dropna(subset=['away_score','home_score']).iterrows():
     try:
      t,p,b=parse_boxscore(int(gr.game_pk),self.mlb.boxscore(int(gr.game_pk)),gr)
      self.repo.upsert_team_stats(t);self.repo.upsert_pitcher_stats(p);self.repo.upsert_batting(b);boxes+=1
     except Exception:continue
   day+=timedelta(days=1)
  return {'games':synced,'boxscores':boxes}
 def datasets(self):
  games=self.repo.completed_games();team=self.repo.query('SELECT * FROM team_game_stats');pitch=self.repo.query('SELECT * FROM pitcher_game_stats');bat=self.repo.query('SELECT * FROM player_game_batting')
  return (
    build_game_features(games, team, pitch),
    build_batter_training(bat, pitch),
    build_pitcher_k_training(pitch),
)
 def train_all(self):
  game,bat,k=self.datasets();out={}
  targets=[('home_win',game,FEATURES,'label',BinaryTimeModel(self.s.calibration_method)),('hit',bat,HIT_FEATURES,'hit',BinaryTimeModel(self.s.calibration_method)),('home_run',bat,HR_FEATURES,'home_run',BinaryTimeModel(self.s.calibration_method)),('strikeouts',k,K_FEATURES,'strikeouts',CountTimeModel())]
  for name,df,features,label,model in targets:
   try:
    m=model.fit(df,features,label);path=self.s.model_dir/f'{name}.joblib';model.save(path,{'features':features,'target':name,'trained_at':datetime.utcnow().isoformat()});out[name]=asdict(m)
   except Exception as exc:out[name]={'error':str(exc),'rows':len(df)}
  return out
 def game_predictions(self,d:date):
  games=self.repo.games_for_date(d)
  if games.empty:return games
  team=self.repo.query('SELECT * FROM team_game_stats');pitch=self.repo.query('SELECT * FROM pitcher_game_stats');f=build_game_features(games,team,pitch);path=self.s.model_dir/'home_win.joblib'
  if not path.exists():return games.assign(model_status='Train the home-win model first')
  model=BinaryTimeModel.load(path);p=model.predict(f[FEATURES]);out=games.copy();out['home_win_probability']=p;out['away_win_probability']=1-p;out['pick']=out.apply(lambda r:r.home_team if r.home_win_probability>=.5 else r.away_team,axis=1);out['confidence']=out[['home_win_probability','away_win_probability']].max(axis=1);return out
