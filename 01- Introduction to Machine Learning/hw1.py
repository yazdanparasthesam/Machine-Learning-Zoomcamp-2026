import pandas as pd, numpy as np
print("Q1 pandas:", pd.__version__)
df = pd.read_csv("car_fuel_efficiency_2026.csv")
print("Q2 records:", len(df))
print("Q3 fuel types:", df.fuel_type.nunique(), df.fuel_type.unique())
nulls = df.isnull().sum()
print(nulls)
print("Q4 cols with missing:", (nulls>0).sum())
print("Q5 max mpg Asia:", df[df.origin=='Asia'].fuel_efficiency_mpg.max())
med = df.horsepower.median(); mode = df.horsepower.mode()[0]
print("Q6 median before:", med, "mode:", mode)
med2 = df.horsepower.fillna(mode).median()
print("Q6 median after:", med2)
asia = df[df.origin=='Asia'][['vehicle_weight','model_year']].head(7)
X = asia.values
XTX = X.T @ X
w = np.linalg.inv(XTX) @ X.T @ np.array([1100,1300,800,900,1000,1100,1200])
print("Q7 w:", w, "sum:", w.sum())
