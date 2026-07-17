import numpy as np, pandas as pd
FEATURES=['home_win_pct_30','away_win_pct_30','home_runs_pg_14','away_runs_pg_14','home_ops_14','away_ops_14','home_sp_era_5','away_sp_era_5','home_sp_kbb_5','away_sp_kbb_5','home_bullpen_pitches_3','away_bullpen_pitches_3','rest_diff','park_factor','temperature','humidity','wind_speed']
def _prior(df,day,n): return df[pd.to_datetime(df.game_date)<pd.Timestamp(day)].sort_values('game_date').tail(n)
def build_game_features(games,team_stats,pitcher_stats):
 rows=[]; ts=team_stats.copy(); ps=pitcher_stats.copy()
 for _,g in games.sort_values('game_date').iterrows():
  day=g.game_date; r={'game_pk':g.game_pk,'game_date':day,'label':float(g.home_score>g.away_score) if pd.notna(g.home_score) and pd.notna(g.away_score) else np.nan}
  for side in ('home','away'):
   tid=g[f'{side}_team_id']; hist=_prior(ts[ts.team_id==tid],day,30); h14=hist.tail(14); h3=hist.tail(3)
   wins=[]
   for _,x in hist.iterrows():
    opp=ts[(ts.game_pk==x.game_pk)&(ts.team_id==x.opponent_id)]
    if not opp.empty:wins.append(float(x.runs>opp.runs.iloc[0]))
   r[f'{side}_win_pct_30']=np.mean(wins) if wins else .5
   r[f'{side}_runs_pg_14']=h14.runs.mean() if len(h14) else 4.3
   obp=((h14.hits.sum()+h14.walks.sum())/(h14.at_bats.sum()+h14.walks.sum())) if h14.at_bats.sum()>0 else .320
   slg=(h14.total_bases.sum()/h14.at_bats.sum()) if h14.at_bats.sum()>0 else .400
   r[f'{side}_ops_14']=obp+slg; r[f'{side}_bullpen_pitches_3']=h3.bullpen_pitches.sum() if len(h3) else 90
   pid=g.get(f'{side}_probable_pitcher_id'); ph=_prior(ps[(ps.player_id==pid)&(ps.games_started>0)],day,5) if pd.notna(pid) else ps.iloc[0:0]
   ip=ph.innings_pitched.sum(); r[f'{side}_sp_era_5']=9*ph.earned_runs.sum()/ip if ip else 4.30; r[f'{side}_sp_kbb_5']=ph.strikeouts.sum()/max(ph.walks.sum(),1) if ip else 2.5
   last=hist.game_date.max() if len(hist) else None; r[f'{side}_rest']=min(max((pd.Timestamp(day)-pd.Timestamp(last)).days,0),7) if last else 3
  r['rest_diff']=r['home_rest']-r['away_rest']; r['park_factor']=g.get('park_factor',1) or 1
  for c,default in [('temperature',72),('humidity',50),('wind_speed',7)]:r[c]=g.get(c,default) if pd.notna(g.get(c)) else default
  rows.append(r)
 return pd.DataFrame(rows)
