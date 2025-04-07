from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 

# train xgboost model
model = XGBClassifier(eval_metric='logloss')
model.fit(X, y)

#print model roc_auc score
y_pred = model.predict(X)
roc_auc = roc_auc_score(y, y_pred)
print(f'Model ROC-AUC: {roc_auc}')