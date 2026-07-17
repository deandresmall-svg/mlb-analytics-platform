import pandas as pd
from mlb_analytics.features.player_features import build_batter_training

def test_batter_features_are_prior_only():
 d=pd.DataFrame([{'game_pk':i,'game_date':f'2026-04-{i:02d}','team_id':1,'player_id':9,'player_name':'P','side':'home','plate_appearances':4,'at_bats':4,'hits':1 if i<5 else 0,'home_runs':0,'walks':0,'strikeouts':1,'total_bases':1} for i in range(1,8)])
 f=build_batter_training(d);assert not f.empty;assert f.iloc[0].hit_rate_10==1.0
