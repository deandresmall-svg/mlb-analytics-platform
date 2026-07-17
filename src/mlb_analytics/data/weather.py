from datetime import date, datetime
from .http import JSONClient
class WeatherClient:
    def __init__(self,forecast_url:str,archive_url:str,timeout=20,retries=3):
        self.forecast=JSONClient(forecast_url,timeout,retries); self.archive=JSONClient(archive_url,timeout,retries)
    def hourly(self,lat:float,lon:float,day:date)->dict:
        vars='temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation'
        params={'latitude':lat,'longitude':lon,'start_date':day.isoformat(),'end_date':day.isoformat(),'hourly':vars,'timezone':'UTC'}
        client=self.archive if day < date.today() else self.forecast
        return client.get('',params)
    @staticmethod
    def nearest(payload:dict, game_time:str|datetime|None)->dict:
        h=payload.get('hourly',{}); times=h.get('time',[])
        if not times:return {}
        target=datetime.fromisoformat(str(game_time).replace('Z','+00:00')).replace(tzinfo=None) if game_time else datetime.fromisoformat(times[0])
        idx=min(range(len(times)),key=lambda i:abs((datetime.fromisoformat(times[i])-target).total_seconds()))
        return {k:(v[idx] if isinstance(v,list) and len(v)>idx else None) for k,v in h.items() if k!='time'}
