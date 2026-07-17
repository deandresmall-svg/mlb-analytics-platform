from datetime import date
from typing import Any
from .http import JSONClient
class MLBClient(JSONClient):
    def schedule(self,start:date,end:date|None=None)->dict[str,Any]:
        p={'sportId':1,'startDate':start.isoformat(),'endDate':(end or start).isoformat(),'hydrate':'probablePitcher,venue,linescore'}
        return self.get('schedule',p)
    def boxscore(self,game_pk:int)->dict[str,Any]: return self.get(f'game/{game_pk}/boxscore')
    def live_feed(self,game_pk:int)->dict[str,Any]: return self.get(f'game/{game_pk}/feed/live')
