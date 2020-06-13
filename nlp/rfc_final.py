import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,\
        precision_score, recall_score, classification_report, f1_score, log_loss

from scipy.spatial.distance import cosine


df_ner = pickle.load(open('uni_enc.pkl', 'rb'))

print("Data set size:", len(df_ner))

df_train, df_test = train_test_split(df_ner, test_size=0.10,random_state=13)

### concatenate q1 and q2 fast_text vectors
X_train = np.concatenate((np.stack(df_train['vectorize_q1'].to_numpy(), axis=0),\
                          np.stack(df_train['vectorize_q2'].to_numpy(), axis=0)), axis=1)
X_test = np.concatenate((np.stack(df_test['vectorize_q1'].to_numpy(), axis=0),\
                          np.stack(df_test['vectorize_q2'].to_numpy(), axis=0)), axis=1)

### Target features
y_test = df_test['is_duplicate']
y_train = df_train['is_duplicate']

### train model only on fasttext vectors
print("Start training RFC")
rfc = RandomForestClassifier(n_estimators = 600,n_jobs = 8, min_samples_split = 10, class_weight = {0:1, 1:2})
rfc.fit(X_train[:100000], y_train[:100000])
y_pred = rfc.predict(X_test)
print('\n clasification report (RFC (200 est) + fast_text_vectors:\n',\
        classification_report(y_test, y_pred))

pickle.dump(rfc, open("rfc_for_probas.pkl", 'wb'))
rfc = pickle.load( open("rfc_for_probas.pkl", 'rb'))

### Calculate predict probas to  calculate log_loss and use them as a feature

rfc_probas_train = rfc.predict_proba(X_train)
rfc_probas_test = rfc.predict_proba(X_test)

log_los = log_loss(y_test, rfc_probas_test)
print('Log loss fof small df:', round(log_los,2))


rfc_probas_train_0 = [x[0] for x in rfc_probas_train]
rfc_probas_test_0 = [x[0] for x in rfc_probas_test]

df_train['rfc_probas_0'] = rfc_probas_train_0
df_test['rfc_probas_0'] = rfc_probas_test_0


### Train final classificator
feats=['ner_score','lev_dist','cos_dist','score','rfc_probas_0']

rfc_c_p = RandomForestClassifier(n_estimators = 100, class_weight = {0:2, 1:10},\
                                 min_samples_split = 8, min_samples_leaf = 2)
rfc_c_p.fit(df_train[feats],y_train)
y_pred_c_p = rfc_c_p.predict(df_test[feats])


print('\n clasification report (RFC cw 2,10 + cosine distance + WER + predict_probas + NER:\n',\
        classification_report(y_test, y_pred_c_p))

rfc_c_p = RandomForestClassifier(n_estimators = 50, class_weight = {0:2, 1:10},\
                                 min_samples_split = 500, min_samples_leaf = 4)
rfc_c_p.fit(df_train[feats],y_train)
y_pred_c_p = rfc_c_p.predict(df_test[feats])


rfc_c_p_probas_test = rfc_c_p.predict_proba(df_test[feats])
log_los_rfc = log_loss(y_test, rfc_c_p_probas_test)
print('Log loss:', round(log_los_rfc,2))

print('\n clasification report (RFC wc 2,10 + cosine distance + WER + predict_probas + NER:\n',\
        classification_report(y_test, y_pred_c_p))


#### XGBOOST
print("XGBoost")
xgb_aut = XGBClassifier(n_estimators=100,eta=0.1, scale_pos_weight =5, eval_metric='auc',max_depth=10)
xgb_aut.fit(df_train[feats],y_train)
y_pred_xgb = xgb_aut.predict(df_test[feats])

xgb_aut_probas_test = xgb_aut.predict_proba(df_test[feats])
log_los_xgb = log_loss(y_test, xgb_aut_probas_test)
print('Log loss:', round(log_los_xgb,2))

print('\n clasification report (XGB wc 3,6 + cosine distance + WER + predict_probas + NER:\n',\
        classification_report(y_test, y_pred_xgb))
print(xgb_aut.feature_importances_)
for i,f in enumerate(feats):
   print(f,xgb_aut.feature_importances_[i])
