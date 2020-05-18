import pandas as pd
import numpy as np

def lev_dist(a1, a2):
    source=a1.split()
    target=a2.split()
    if source == target:
        return 0

    # Prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    for i in range(slen+1):
        dist[i][0] = i
    for j in range(tlen+1):
        dist[0][j] = j

    # Counting distance, here is my function
    for i in range(slen):
        for j in range(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i+1][j+1] = min(
                            dist[i][j+1] + 1,   # deletion
                            dist[i+1][j] + 1,   # insertion
                            dist[i][j] + cost   # substitution
                        )
    return (dist[-1][-1])/\
           ((len(source)+len(target))/2)



df = pd.read_csv('preprocess_all.csv').dropna()
df['lev_dist'] = np.vectorize(lev_dist)(df['preprocessed_q1'], df['preprocessed_q2'])
df.to_csv('preprocess_all_lev.csv')
df['lev_pred']=0
df['lev_pred']=df.lev_dist < 0.35
df["lev_pred"]=df["lev_pred"].astype(int)
df['lev_true']=df.lev_pred ==df.is_duplicate
print(df['lev_true'].value_counts())


print(len(df))
