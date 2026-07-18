from dataclasses import asdict
from datetime import date,timedelta,datetime
import pandas as pd
from mlb_analytics.config import Settings
from mlb_analytics.data.mlb_api import MLBClient
from mlb_analytics.data.weather import WeatherClient
from mlb_analytics.data.venues import venue_info
from mlb_analytics.data.parsers import parse_schedule,parse_boxscore
from mlb_analytics.data.repository import Repository
from mlb_analytics.data.odds_api import OddsAPIClient, MLB_PROP_MARKETS, normalize_name
from mlb_analytics.features.game_features import FEATURES,build_game_features
from mlb_analytics.features.player_features import (
 HIT_FEATURES,HR_FEATURES,K_FEATURES,build_batter_training,
 build_pitcher_k_training,build_batter_prediction_rows,
 build_pitcher_prediction_row
)
from mlb_analytics.models.base import BinaryTimeModel,CountTimeModel
class AnalyticsService:
 def __init__(self,s:Settings):
  self.s=s;s.ensure_directories();self.mlb=MLBClient(s.api_base_url,s.api_timeout_seconds,s.api_max_retries);self.weather=WeatherClient(s.weather_base_url,s.weather_archive_url,s.api_timeout_seconds,s.api_max_retries);self.repo=Repository(s.database_url);self.repo.initialize();self.odds=OddsAPIClient(s.odds_api_key,s.odds_api_base_url,s.api_timeout_seconds)
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
 def batter_predictions_for_game(self,game_row):
  bat=self.repo.query('SELECT * FROM player_game_batting')
  pitch=self.repo.query('SELECT * FROM pitcher_game_stats')
  game_date=pd.Timestamp(game_row['game_date']).date()
  outputs=[]
  for side in ('away','home'):
   rows=build_batter_prediction_rows(
    bat,pitch,int(game_row[f'{side}_team_id']),
    game_row.get(f'{"home" if side=="away" else "away"}_probable_pitcher_id'),
    game_date,side
   )
   if rows.empty:continue
   rows['team']=game_row[f'{side}_team']
   rows['opponent_pitcher']=game_row.get(
    f'{"home" if side=="away" else "away"}_probable_pitcher','TBD'
   )
   for target,features in [('hit',HIT_FEATURES),('home_run',HR_FEATURES)]:
    path=self.s.model_dir/f'{target}.joblib'
    if path.exists():
     model=BinaryTimeModel.load(path)
     rows[f'{target}_probability']=model.predict(rows[features])
   outputs.append(rows)
  if not outputs:return pd.DataFrame()
  out=pd.concat(outputs,ignore_index=True)
  for target in ('hit','home_run'):
   col=f'{target}_probability'
   if col in out.columns:
    out[f'{target}_confidence']=out[col].map(self._confidence_label)
  return out

 def pitcher_predictions_for_game(self,game_row):
  pitch=self.repo.query('SELECT * FROM pitcher_game_stats')
  path=self.s.model_dir/'strikeouts.joblib'
  if not path.exists():return pd.DataFrame()
  model=CountTimeModel.load(path);rows=[]
  game_date=pd.Timestamp(game_row['game_date']).date()
  for side in ('away','home'):
   frame=build_pitcher_prediction_row(
    pitch,game_row.get(f'{side}_probable_pitcher_id'),game_date
   )
   if frame.empty:continue
   frame['team']=game_row[f'{side}_team']
   frame['opponent']=game_row[f'{"home" if side=="away" else "away"}_team']
   frame['projected_strikeouts']=model.predict(frame[K_FEATURES])
   rows.append(frame)
  return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

 def odds_for_slate(self,d:date):
  if not self.s.odds_api_key:
   return {"rows":pd.DataFrame(),"usage":{},"error":"ODDS_API_KEY is not configured","event_errors":[],"matched_games":0,"requested_games":0}
  games=self.repo.games_for_date(d)
  if games.empty:
   return {"rows":pd.DataFrame(),"usage":{},"error":"No MLB games are stored for the selected date. Sync the schedule first.","event_errors":[],"matched_games":0,"requested_games":0}
  try:
   events=self.odds.events_for_date(d)
  except Exception as exc:
   return {"rows":pd.DataFrame(),"usage":self.odds.usage.__dict__,"error":str(exc),"event_errors":[],"matched_games":0,"requested_games":0}
  frames=[];event_errors=[];matched=0;total_cost=0
  for _,game in games.iterrows():
   event=self.odds.match_event(game,events)
   if not event:
    event_errors.append(f"No Odds API event match for {game.get('away_team')} at {game.get('home_team')}")
    continue
   matched+=1
   try:
    payload=self.odds.event_odds(event["id"],MLB_PROP_MARKETS,self.s.odds_regions,self.s.odds_bookmakers or None)
    total_cost += self.odds.usage.last or 0
    frame=self.odds.flatten_props(payload)
    if not frame.empty:
     frame["game_pk"]=game.get("game_pk");frames.append(frame)
   except Exception as exc:
    event_errors.append(f"{game.get('away_team')} at {game.get('home_team')}: {exc}")
  rows=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
  usage=self.odds.usage.__dict__.copy();usage["refresh_cost"]=total_cost
  return {"rows":rows,"usage":usage,"event_errors":event_errors,"matched_games":matched,"requested_games":len(games)}

 @staticmethod
 def _confidence_label(probability):
  if pd.isna(probability):return 'Unavailable'
  if probability>=.68:return 'High'
  if probability>=.58:return 'Medium'
  return 'Low'


 def slate_prop_predictions(self,d:date):
  games=self.repo.games_for_date(d)
  if games.empty:return {'batters':pd.DataFrame(),'pitchers':pd.DataFrame()}
  batter_frames=[];pitcher_frames=[]
  for _,game in games.iterrows():
   try:
    batters=self.batter_predictions_for_game(game)
    if not batters.empty:
     batters=batters.copy();batters['game_pk']=game.get('game_pk');batters['matchup']=f"{game.get('away_team')} at {game.get('home_team')}";batters['game_time']=game.get('game_time');batter_frames.append(batters)
   except Exception:pass
   try:
    pitchers=self.pitcher_predictions_for_game(game)
    if not pitchers.empty:
     pitchers=pitchers.copy();pitchers['game_pk']=game.get('game_pk');pitchers['matchup']=f"{game.get('away_team')} at {game.get('home_team')}";pitchers['game_time']=game.get('game_time');pitcher_frames.append(pitchers)
   except Exception:pass
  return {
   'batters':pd.concat(batter_frames,ignore_index=True) if batter_frames else pd.DataFrame(),
   'pitchers':pd.concat(pitcher_frames,ignore_index=True) if pitcher_frames else pd.DataFrame(),
  }
