import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, accuracy_score

df = pd.read_csv('course_lead_scoring_2026.csv')
cat = list(df.dtypes[df.dtypes == 'object'].index)
num = [c for c in df.columns if c not in cat + ['converted']]
df[cat] = df[cat].fillna('NA')
df[num] = df[num].fillna(0.0)

print("Q1 mode industry:", df.industry.mode()[0])
print(df.industry.value_counts().head())

corr = df[num].corr()
print("\ncorr:\n", corr.round(4))
pairs = [('interaction_count','lead_score'),('number_of_courses_viewed','lead_score'),
         ('number_of_courses_viewed','interaction_count'),('annual_income','interaction_count')]
for a,b in pairs: print("Q2", a, b, round(corr.loc[a,b],4), "abs", round(abs(corr.loc[a,b]),4))

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True); df_val = df_val.reset_index(drop=True); df_test = df_test.reset_index(drop=True)
y_train = df_train.converted.values; y_val = df_val.converted.values; y_test = df_test.converted.values
for d in (df_train, df_val, df_test): del d['converted']

print("\nQ3 MI:")
for c in cat: print(" ", c, round(mutual_info_score(df_train[c], y_train),2), mutual_info_score(df_train[c], y_train))

features = cat + num
def train(feats, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_tr = dv.fit_transform(df_train[feats].to_dict(orient='records'))
    X_v = dv.transform(df_val[feats].to_dict(orient='records'))
    m = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42).fit(X_tr, y_train)
    return accuracy_score(y_val, m.predict(X_v))

acc = train(features)
print("\nQ4 accuracy:", acc, round(acc,2))

print("\nQ5:")
res={}
for f in features:
    a = train([x for x in features if x != f])
    res[f] = acc - a
    print(f"  {f}: acc_without={a:.5f} diff={acc-a:+.5f}")
for f in ['lead_source','number_of_courses_viewed','interaction_count']:
    print("  candidate", f, round(res[f],5), "abs", abs(res[f]))
print("  smallest diff among candidates:", min(['lead_source','number_of_courses_viewed','interaction_count'], key=lambda f: res[f]))
print("  smallest |diff|:", min(['lead_source','number_of_courses_viewed','interaction_count'], key=lambda f: abs(res[f])))

print("\nQ6:")
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    a = train(features, C)
    print(" ", C, round(a,3), a)
