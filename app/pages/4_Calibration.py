import numpy as np,pandas as pd,streamlit as st
st.title('Calibration Reliability')
st.info('After predictions are persisted, bucket predicted probabilities and compare them with observed outcomes here. The training code already uses chronological holdout calibration; this page is ready for persisted forecast history.')
example=pd.DataFrame({'bucket':['50–55%','55–60%','60–65%','65–70%','70%+'],'predicted':[.525,.575,.625,.675,.74],'observed':[np.nan]*5,'count':[0]*5});st.dataframe(example,use_container_width=True,hide_index=True)
