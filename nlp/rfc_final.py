import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,\
        precision_score, recall_score, classification_report, f1_score, log_loss

from scipy.spatial.distance import cosine


df_ner = pickle.load(open('ner.pkl', 'rb'))

print("Data set size:", len(df_ner))

df_train, df_test = train_test_split(df_ner, test_size=0.10)

### concatenate q1 and q2 fast_text vectors
X_train = np.concatenate((np.stack(df_train['vectorize_q1'].to_numpy(), axis=0),\
                          np.stack(df_train['vectorize_q2'].to_numpy(), axis=0)), axis=1)
X_test = np.concatenate((np.stack(df_test['vectorize_q1'].to_numpy(), axis=0),\
                          np.stack(df_test['vectorize_q2'].to_numpy(), axis=0)), axis=1)

### Target features
y_test = df_test['is_duplicate']
y_train = df_train['is_duplicate']

### train model only on fasttext vectors
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print('\n clasification report (RFC (200 est) + fast_text_vectors:\n',\
        classification_report(y_test, y_pred))

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

rfc_c_p = RandomForestClassifier(n_estimators = 200, class_weight = {0:2, 1:10},\
                                 min_samples_split = 6, min_samples_leaf = 2)
rfc_c_p.fit(df_train[['lev_dist','cos_dist', 'rfc_probas_0']],y_train)
y_pred_c_p = rfc_c_p.predict(df_test[['lev_dist','cos_dist', 'rfc_probas_0']])


print('\n clasification report (RFC + cosine distance + WER + predict_probas + NER:\n',\
        classification_report(y_test, y_pred_c_p))