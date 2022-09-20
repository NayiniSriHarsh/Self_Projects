import pickle
import pandas as pd
def ProcessX(X):
    sm = {'low': 1,
      'medium':2,
      'high':3}
    X['salary'] = X['salary'].map(sm)
    X = pd.get_dummies(X,['Department'])
    train_cols=['salary', 'average_montly_hours', 'time_spend_company', 'Work_accident',
       'last_evaluation', 'satisfaction_level', 'number_project',
       'promotion_last_5years', 'Department_IT', 'Department_RandD',
       'Department_accounting', 'Department_hr', 'Department_management',
       'Department_marketing', 'Department_product_mng', 'Department_sales',
       'Department_support', 'Department_technical']
    missing_cols = set( train_cols ) - set( X.columns )
    for c in missing_cols:
        X[c] = 0
    X = X[train_cols]
    sc = pickle.load(open("scalar.std","rb"))
    return pd.DataFrame(sc.transform(X))