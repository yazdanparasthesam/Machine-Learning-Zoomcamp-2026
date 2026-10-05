# FAQ proposal draft — ML Zoomcamp 2026, Homework 2

Submit via: https://github.com/DataTalksClub/faq/issues/new?template=faq-proposal.yml
Course field: Machine Learning Zoomcamp

---

## Option A (recommended)

**Issue title**
```
FAQ: Homework 2 Q3 - why do fill-with-0 and fill-with-mean look identical?
```

**Question field**
```
Homework 2 Q3: filling the horsepower NAs with 0 and with the mean gives me the same RMSE - how do I tell which option is better?
```

**Answer field**
They are not actually identical, the difference just hides behind rounding. In the pinned 2026
release the validation RMSEs are:

```python
round(rmse_zero, 3)   # 2.205
round(rmse_mean, 3)   # 2.202   <- better
```

Round to 3 decimals as the homework asks (the 2026 wording says this explicitly because 2 decimals
turned it into a tie). So the answer is "With mean".

Two things that silently break this question:

1. The mean must be computed on the **training split only**, never on the full dataset - otherwise
   you leak information from validation/test:

   ```python
   hp_mean = df_train.horsepower.mean()   # correct
   hp_mean = df.horsepower.mean()         # leakage
   ```

2. Use the *same* fill value on train, validation and test. Don't recompute the mean per split.

```python
def prepare_X(d, fill_value):
    return d[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].fillna(fill_value).values
```

---

## Option B

**Issue title**
```
FAQ: Homework 2 Q4 - r=0 beats every regularization value, is that expected?
```

**Question field**
```
Homework 2 Q4: the best RMSE comes from r=0, so regularization only makes my model worse. Is that correct?
```

**Answer field**
Yes, that is the expected result for this homework. With the 2026 dataset and only four numeric
features, X.T @ X is well-conditioned, so the ridge penalty adds bias without removing any
variance problem. Validation RMSE rounded to 4 decimals:

```
r=0     2.2053   <- best
r=0.01  2.2058
r=0.1   2.2241
r=1     2.3492
r=5     2.4094
r=10    2.4195
r=100   2.4292
```

Round to 4 decimals (not 2, not 3), otherwise r=0 and r=0.01 collapse into a tie. The homework
also says: if several values tie for the best score, pick the smallest r.

Regularization helps when features are collinear or nearly duplicated - for example after
one-hot encoding many categories, which is where you will see it matter later in the course.

---

## Option C

**Issue title**
```
FAQ: Homework 2 - my RMSE values differ from everyone else's after the split
```

**Question field**
```
Homework 2: my RMSE numbers don't match the answer options even though my code looks right. What usually causes this?
```

**Answer field**
Almost always one of these four:

1. **Wrong dataset.** Use the pinned 2026 file, not the previous cohort's `car_fuel_efficiency.csv`:

   ```bash
   wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
   ```

2. **Seed set in the wrong place.** `np.random.seed(seed)` must come immediately before
   `np.random.shuffle(idx)`, inside the split, every time. If you shuffle twice without reseeding
   you get a different split.

3. **Using `df.reset_index()` or `sort` before splitting.** The split must index the filtered
   DataFrame exactly as it was read, with `df.iloc[idx[...]]`.

4. **Forgetting to drop the target from X.** `fuel_efficiency_mpg` must not be among the four
   feature columns, otherwise RMSE collapses toward 0.

Reference values for the 2026 release: Q3 = 2.205 / 2.202, Q4 best r = 0 (2.2053),
Q5 std = 0.029, Q6 test RMSE = 2.236.

---

## Option D

**Issue title**
```
FAQ: Homework 2 Q6 - should I retrain on train+validation before scoring the test set?
```

**Question field**
```
Homework 2 Q6: do I reuse the model trained on the training split, or retrain on train + validation?
```

**Answer field**
Retrain. Q6 asks you to combine the train and validation splits into one full-training set and fit
a fresh model on it with r=0.001, then score once on the untouched test split:

```python
df_train, df_val, df_test = split(df, 9)

df_full_train = pd.concat([df_train, df_val])
y_full_train = df_full_train.fuel_efficiency_mpg.values

w0, w = train_linear_regression_reg(prepare_X(df_full_train, 0), y_full_train, r=0.001)
rmse(df_test.fuel_efficiency_mpg.values, prepare_X(df_test, 0).dot(w) + w0)   # 2.236
```

This is the standard pattern from the lectures: validation is used to *choose* the model and its
hyperparameters, and once the choice is made you refit on all the data you are allowed to train on
(60% + 20% = 80%) so the final model sees more examples. The test set is touched exactly once.

Note `pd.concat([df_train, df_val])` keeps the original shuffled index - that is fine, since
`prepare_X` only reads column values.
