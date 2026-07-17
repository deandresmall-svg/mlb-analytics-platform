from mlb_analytics.data.parsers import parse_schedule

def test_parse_schedule():
 p={"dates":[{"games":[{"gamePk":1,"officialDate":"2026-07-17","gameDate":"2026-07-17T23:10:00Z","status":{"detailedState":"Scheduled"},"venue":{"name":"Fenway Park"},"teams":{"away":{"team":{"id":1,"name":"A"}},"home":{"team":{"id":2,"name":"H"}}}}]}]}
 d=parse_schedule(p);assert len(d)==1;assert d.iloc[0].game_pk==1
