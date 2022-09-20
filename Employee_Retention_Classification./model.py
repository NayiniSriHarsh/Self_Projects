import pickle
import pandas as pd
import numpy as np
def Predict(X):
    probf = (lr.predict_proba(X)+rf.predict_proba(X)+xgb.predict_proba(X)+ada.predict_proba(X)+dt.predict_proba(X))/5
    return np.array(pd.DataFrame(probf[:,1])[0].apply(lambda x: 1 if x>=0.5 else 0))

lr = pickle.load(open("lr_model.pkl","rb"))
rf = pickle.load(open("rf_model.pkl","rb"))
dt = pickle.load(open("dt_model.pkl","rb"))
xgb = pickle.load(open("xgb_model.pkl","rb"))
ada = pickle.load(open("ada_model.pkl","rb"))
