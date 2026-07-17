from typing import Any
import time, requests
class APIError(RuntimeError): pass
class JSONClient:
    def __init__(self, base_url:str, timeout:float=20, retries:int=3):
        self.base_url=base_url.rstrip('/'); self.timeout=timeout; self.retries=retries
        self.session=requests.Session(); self.session.headers.update({'Accept':'application/json','User-Agent':'mlb-github-streamlit-analytics/1.0'})
    def get(self,path:str='',params:dict[str,Any]|None=None)->dict[str,Any]:
        url=f"{self.base_url}/{path.lstrip('/')}" if path else self.base_url
        last=None
        for attempt in range(self.retries):
            try:
                r=self.session.get(url,params=params,timeout=self.timeout); r.raise_for_status(); data=r.json()
                if not isinstance(data,dict): raise APIError('Expected a JSON object')
                return data
            except (requests.RequestException,ValueError,APIError) as exc:
                last=exc
                if attempt+1<self.retries: time.sleep(.5*(attempt+1))
        raise APIError(f"Request failed for {url}: {last}")
