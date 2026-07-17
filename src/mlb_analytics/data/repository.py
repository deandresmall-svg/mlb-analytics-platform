import pandas as pd
from sqlalchemy import create_engine, text
SCHEMA = [
"""CREATE TABLE IF NOT EXISTS games(game_pk INTEGER PRIMARY KEY,game_date TEXT,game_time TEXT,status TEXT,away_team_id INTEGER,away_team TEXT,home_team_id INTEGER,home_team TEXT,away_score REAL,home_score REAL,venue TEXT,away_probable_pitcher_id INTEGER,away_probable_pitcher TEXT,home_probable_pitcher_id INTEGER,home_probable_pitcher TEXT,temperature REAL,humidity REAL,pressure REAL,wind_speed REAL,wind_direction REAL,precipitation REAL,park_factor REAL,updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""",
"""CREATE TABLE IF NOT EXISTS team_game_stats(game_pk INTEGER,game_date TEXT,team_id INTEGER,side TEXT,opponent_id INTEGER,runs REAL,hits REAL,home_runs REAL,walks REAL,strikeouts REAL,at_bats REAL,total_bases REAL,bullpen_innings REAL,bullpen_pitches REAL,PRIMARY KEY(game_pk,team_id))""",
"""CREATE TABLE IF NOT EXISTS pitcher_game_stats(game_pk INTEGER,game_date TEXT,team_id INTEGER,player_id INTEGER,player_name TEXT,side TEXT,games_started REAL,innings_pitched REAL,pitches_thrown REAL,hits REAL,runs REAL,earned_runs REAL,walks REAL,strikeouts REAL,home_runs REAL,PRIMARY KEY(game_pk,player_id))""",
"""CREATE TABLE IF NOT EXISTS player_game_batting(game_pk INTEGER,game_date TEXT,team_id INTEGER,player_id INTEGER,player_name TEXT,side TEXT,plate_appearances REAL,at_bats REAL,hits REAL,home_runs REAL,walks REAL,strikeouts REAL,total_bases REAL,PRIMARY KEY(game_pk,player_id))""",
"""CREATE TABLE IF NOT EXISTS model_runs(id INTEGER PRIMARY KEY AUTOINCREMENT,target TEXT,model_version TEXT,trained_at TEXT,rows INTEGER,brier REAL,roc_auc REAL,log_loss REAL,accuracy REAL,mae REAL,artifact_path TEXT)"""
]
class Repository:
 def __init__(self,url): self.engine=create_engine(url,future=True)
 def initialize(self):
  with self.engine.begin() as c:
   for q in SCHEMA:c.execute(text(q))
 def _upsert(self,table,df,keys):
  if df.empty:return 0
  cols=list(df.columns); upd=[c for c in cols if c not in keys]
  q=text(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join(':'+c for c in cols)}) ON CONFLICT({','.join(keys)}) DO UPDATE SET "+','.join(f"{c}=excluded.{c}" for c in upd))
  x=df.copy()
  for c in x.columns:x[c]=x[c].where(x[c].notna(),None)
  with self.engine.begin() as conn:conn.execute(q,x.to_dict('records'))
  return len(x)
 def upsert_games(self,df): return self._upsert('games',df,['game_pk'])
 def upsert_team_stats(self,df): return self._upsert('team_game_stats',df,['game_pk','team_id'])
 def upsert_pitcher_stats(self,df): return self._upsert('pitcher_game_stats',df,['game_pk','player_id'])
 def upsert_batting(self,df): return self._upsert('player_game_batting',df,['game_pk','player_id'])
 def query(self,q,params=None): return pd.read_sql(text(q),self.engine,params=params or {})
 def games_for_date(self,d): return self.query('SELECT * FROM games WHERE game_date=:d ORDER BY game_time',{'d':d.isoformat()})
 def games_between(self,start,end): return self.query('SELECT * FROM games WHERE game_date BETWEEN :s AND :e ORDER BY game_date,game_time',{'s':start.isoformat(),'e':end.isoformat()})
 def completed_games(self): return self.query("SELECT * FROM games WHERE home_score IS NOT NULL AND away_score IS NOT NULL")
 def coverage(self): return self.query('SELECT (SELECT COUNT(*) FROM games) games,(SELECT COUNT(*) FROM team_game_stats) team_rows,(SELECT COUNT(*) FROM pitcher_game_stats) pitcher_rows,(SELECT COUNT(*) FROM player_game_batting) batter_rows')
