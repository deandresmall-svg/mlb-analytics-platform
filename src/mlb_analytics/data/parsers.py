from datetime import datetime
import pandas as pd

def parse_schedule(payload):
 rows=[]
 for d in payload.get('dates',[]):
  for g in d.get('games',[]):
   t=g.get('teams',{}); a=t.get('away',{}); h=t.get('home',{}); ap=a.get('probablePitcher',{}); hp=h.get('probablePitcher',{})
   rows.append({'game_pk':g.get('gamePk'),'game_date':g.get('officialDate'),'game_time':g.get('gameDate'),'status':g.get('status',{}).get('detailedState'),
   'away_team_id':a.get('team',{}).get('id'),'away_team':a.get('team',{}).get('name'),'home_team_id':h.get('team',{}).get('id'),'home_team':h.get('team',{}).get('name'),
   'away_score':a.get('score'),'home_score':h.get('score'),'venue':g.get('venue',{}).get('name'),'away_probable_pitcher_id':ap.get('id'),'away_probable_pitcher':ap.get('fullName'),'home_probable_pitcher_id':hp.get('id'),'home_probable_pitcher':hp.get('fullName')})
 return pd.DataFrame(rows)

def _num(v):
 try:return float(v)
 except (TypeError,ValueError):return None

def parse_boxscore(game_pk,payload,games_row):
 team_rows=[]; pitcher_rows=[]; player_rows=[]
 teams=payload.get('teams',{})
 for side in ('away','home'):
  block=teams.get(side,{}); stats=block.get('teamStats',{}); bat=stats.get('batting',{}); pit=stats.get('pitching',{})
  team_id=games_row[f'{side}_team_id']
  team_rows.append({'game_pk':game_pk,'game_date':games_row['game_date'],'team_id':team_id,'side':side,'opponent_id':games_row['home_team_id' if side=='away' else 'away_team_id'],
   'runs':games_row[f'{side}_score'],'hits':_num(bat.get('hits')),'home_runs':_num(bat.get('homeRuns')),'walks':_num(bat.get('baseOnBalls')),'strikeouts':_num(bat.get('strikeOuts')),
   'at_bats':_num(bat.get('atBats')),'total_bases':_num(bat.get('totalBases')),'bullpen_innings':0.0,'bullpen_pitches':0.0})
  for p in block.get('players',{}).values():
   person=p.get('person',{}); s=p.get('stats',{}); ps=s.get('pitching',{}); bs=s.get('batting',{})
   if ps:
    starter=bool(p.get('gameStatus',{}).get('isCurrentPitcher')) or p.get('position',{}).get('abbreviation')=='P' and len(pitcher_rows)==0
    pitcher_rows.append({'game_pk':game_pk,'game_date':games_row['game_date'],'team_id':team_id,'player_id':person.get('id'),'player_name':person.get('fullName'),'side':side,
     'games_started':_num(ps.get('gamesStarted')) or 0,'innings_pitched':_num(ps.get('inningsPitched')) or 0,'pitches_thrown':_num(ps.get('numberOfPitches')) or 0,'hits':_num(ps.get('hits')) or 0,
     'runs':_num(ps.get('runs')) or 0,'earned_runs':_num(ps.get('earnedRuns')) or 0,'walks':_num(ps.get('baseOnBalls')) or 0,'strikeouts':_num(ps.get('strikeOuts')) or 0,'home_runs':_num(ps.get('homeRuns')) or 0})
   if bs and (_num(bs.get('plateAppearances')) or 0)>0:
    player_rows.append({'game_pk':game_pk,'game_date':games_row['game_date'],'team_id':team_id,'player_id':person.get('id'),'player_name':person.get('fullName'),'side':side,
     'plate_appearances':_num(bs.get('plateAppearances')) or 0,'at_bats':_num(bs.get('atBats')) or 0,'hits':_num(bs.get('hits')) or 0,'home_runs':_num(bs.get('homeRuns')) or 0,'walks':_num(bs.get('baseOnBalls')) or 0,'strikeouts':_num(bs.get('strikeOuts')) or 0,'total_bases':_num(bs.get('totalBases')) or 0})
 # bullpen = all pitchers with games_started == 0
 for tr in team_rows:
  rel=[p for p in pitcher_rows if p['team_id']==tr['team_id'] and p['games_started']==0]
  tr['bullpen_innings']=sum(p['innings_pitched'] for p in rel); tr['bullpen_pitches']=sum(p['pitches_thrown'] for p in rel)
 return pd.DataFrame(team_rows),pd.DataFrame(pitcher_rows),pd.DataFrame(player_rows)
