from dataclasses import dataclass
from pathlib import Path
import joblib,numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,PoissonRegressor
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
@dataclass
class Metrics: rows:int; brier:float|None=None; roc_auc:float|None=None; log_loss:float|None=None; accuracy:float|None=None; mae:float|None=None
class BinaryTimeModel:
 def __init__(self,method='sigmoid'):
  base=Pipeline([('impute',SimpleImputer()),('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,class_weight='balanced'))]); self.model=CalibratedClassifierCV(base,method=method,cv=3)
 def fit(self,df,features,label):
  d=df.dropna(subset=[label]).sort_values('game_date'); cut=max(int(len(d)*.8),1); tr,te=d.iloc[:cut],d.iloc[cut:]
  if len(te)<10:raise ValueError('Not enough chronological holdout rows')
  self.model.fit(tr[features],tr[label]); p=self.model.predict_proba(te[features])[:,1]
  return Metrics(len(d),brier_score_loss(te[label],p),roc_auc_score(te[label],p) if te[label].nunique()>1 else None,log_loss(te[label],p),accuracy_score(te[label],p>=.5))
 def predict(self,x):return self.model.predict_proba(x)[:,1]
 def save(self,path,meta):Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump({'model':self.model,'meta':meta},path)
 @classmethod
 def load(cls,path):o=cls();d=joblib.load(path);o.model=d['model'];return o
class CountTimeModel:
 def __init__(self):self.model=Pipeline([('impute',SimpleImputer()),('scale',StandardScaler()),('model',PoissonRegressor(alpha=.2,max_iter=2000))])
 def fit(self,df,features,label):
  d=df.dropna(subset=[label]).sort_values('game_date');cut=max(int(len(d)*.8),1);tr,te=d.iloc[:cut],d.iloc[cut:]
  if len(te)<10:raise ValueError('Not enough chronological holdout rows')
  self.model.fit(tr[features],tr[label]);p=np.maximum(self.model.predict(te[features]),0);return Metrics(len(d),mae=mean_absolute_error(te[label],p))
 def predict(self,x):return np.maximum(self.model.predict(x),0)
 def save(self,path,meta):Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump({'model':self.model,'meta':meta},path)
