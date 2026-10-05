import pandas as pd, numpy as np

cols = ['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']
df = pd.read_csv('car_fuel_efficiency_2026.csv')[cols]

print("Q1 nulls:\n", df.isnull().sum())
print("Q2 median horsepower:", df.horsepower.median())
print("skew (long tail?):", df.fuel_efficiency_mpg.skew())

def split(df, seed):
    n = len(df); n_val = int(n*0.2); n_test = int(n*0.2); n_train = n - n_val - n_test
    np.random.seed(seed)
    idx = np.arange(n); np.random.shuffle(idx)
    return (df.iloc[idx[:n_train]], df.iloc[idx[n_train:n_train+n_val]], df.iloc[idx[n_train+n_val:]])

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0]); X = np.column_stack([ones, X])
    XTX = X.T.dot(X); w = np.linalg.inv(XTX).dot(X.T).dot(y)
    return w[0], w[1:]

def train_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0]); X = np.column_stack([ones, X])
    XTX = X.T.dot(X) + r*np.eye(X.shape[1])
    w = np.linalg.inv(XTX).dot(X.T).dot(y)
    return w[0], w[1:]

def rmse(y, yp): return np.sqrt(((y-yp)**2).mean())

base = ['engine_displacement','horsepower','vehicle_weight','model_year']
def prep(d, fill):
    X = d[base].fillna(fill)
    return X.values

# Q3
tr, va, te = split(df, 42)
y_tr = tr.fuel_efficiency_mpg.values; y_va = va.fuel_efficiency_mpg.values
for name, fill in [('0', 0), ('mean', tr.horsepower.mean())]:
    w0, w = train_linear_regression(prep(tr, fill), y_tr)
    s = rmse(y_va, prep(va, fill).dot(w) + w0)
    print(f"Q3 fill {name}: {round(s,3)}  (raw {s})")

# Q4
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_reg(prep(tr,0), y_tr, r)
    s = rmse(y_va, prep(va,0).dot(w)+w0)
    print(f"Q4 r={r}: {round(s,4)}")

# Q5
scores=[]
for seed in range(10):
    t,v,_ = split(df, seed)
    w0,w = train_linear_regression(prep(t,0), t.fuel_efficiency_mpg.values)
    scores.append(rmse(v.fuel_efficiency_mpg.values, prep(v,0).dot(w)+w0))
print("Q5 scores:", [round(s,4) for s in scores], "std:", round(np.std(scores),3), np.std(scores))

# Q6
t,v,te = split(df, 9)
full = pd.concat([t,v])
w0,w = train_reg(prep(full,0), full.fuel_efficiency_mpg.values, 0.001)
print("Q6 test rmse:", rmse(te.fuel_efficiency_mpg.values, prep(te,0).dot(w)+w0))
