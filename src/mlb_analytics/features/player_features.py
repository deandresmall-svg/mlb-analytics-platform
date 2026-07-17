import pandas as pd
HIT_FEATURES=['hit_rate_10','hit_rate_30','pa_10','home_run_rate_30','strikeout_rate_30']
HR_FEATURES=['home_run_rate_10','home_run_rate_30','hits_30','total_bases_30']
K_FEATURES=['pitcher_k_per_ip_5','pitcher_pitches_5','pitcher_ip_5']
def build_batter_training(batting):
 rows=[]; d=batting.sort_values(['player_id','game_date'])
 for pid,g in d.groupby('player_id'):
  g=g.reset_index(drop=True)
  for i,r in g.iterrows():
   p10=g.iloc[max(0,i-10):i]; p30=g.iloc[max(0,i-30):i]
   if len(p10)<3:continue
   rows.append({'game_pk':r.game_pk,'player_id':pid,'game_date':r.game_date,'hit':float(r.hits>0),'home_run':float(r.home_runs>0),'hit_rate_10':(p10.hits>0).mean(),'hit_rate_30':(p30.hits>0).mean(),'pa_10':p10.plate_appearances.mean(),'home_run_rate_10':(p10.home_runs>0).mean(),'home_run_rate_30':(p30.home_runs>0).mean(),'strikeout_rate_30':p30.strikeouts.sum()/max(p30.plate_appearances.sum(),1),'hits_30':p30.hits.sum(),'total_bases_30':p30.total_bases.sum()})
 return pd.DataFrame(rows)
def build_pitcher_k_training(pitching):
 rows=[]; d=pitching[pitching.games_started>0].sort_values(['player_id','game_date'])
 for pid,g in d.groupby('player_id'):
  g=g.reset_index(drop=True)
  for i,r in g.iterrows():
   p=g.iloc[max(0,i-5):i]
   if len(p)<2:continue
   ip=p.innings_pitched.sum(); rows.append({'game_pk':r.game_pk,'player_id':pid,'game_date':r.game_date,'strikeouts':r.strikeouts,'pitcher_k_per_ip_5':p.strikeouts.sum()/max(ip,1),'pitcher_pitches_5':p.pitches_thrown.mean(),'pitcher_ip_5':p.innings_pitched.mean()})
 return pd.DataFrame(rows)
