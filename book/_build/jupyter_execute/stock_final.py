# Final

My final prediction model is build on the transformed data set from the previous section. That is, I ignore that the time index contains special information and act as if it were only another features. In the following I consider three models.

1. Linear model
2. Two-stage linear model (*the final model*)

Again, if you only care about the final model please jump directly to subsection 2.

## Preliminaries

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor

ROOT = Path(os.getcwd()).parent

df_train = pd.read_parquet(ROOT / "bld" / "train_stock.parquet")
df_val = pd.read_parquet(ROOT / "bld" / "validate_stock.parquet")

y_train = df_train["Y"]
X_train = df_train.drop("Y", axis=1)

y_val = df_val["Y"]
X_val = df_val.drop("Y", axis=1)

## 1. Linear Model

Here I fit a simple unregularized linear model which is used as a lower benchmark.

lm = LinearRegression()
lm.fit(X_train, y_train)

prediction = lm.predict(X_val)
mse_lm = mean_squared_error(y_val, prediction)
print(f"(Linear Model) MSE: {mse_lm}")

## 2. Two-stage Linear Model (*final model*)

To mix things up, here I select features using a Lasso approach. With these features I then fit a simple 2nd degree polynomial model. The code which I used to construct the final predictions can be found in the script [final_prediction.py](https://github.com/timmens/topics-project/blob/main/codes/final_prediction.py).

***Lasso feature selection***

The regularization parameter is selected via a 5-fold cross-validation procedure over a logspace grid (a sequence which is linear on a logarithmic scale). I select all columns which have nonzero coefficients.

lasso_model = LassoCV(alphas=np.logspace(-2.5, 1, 50), cv=5)
lasso_model = lasso_model.fit(X_train, y_train)

relevant = X_train.columns[lasso_model.coef_ != 0].to_list()
print("Relevant features chosen via Lasso:")
print(relevant)

***Polynomial regression on subset***

def make_features(X, relevant_columns):
    """Return 2nd degree polynomial features plus third power."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    XX = poly.fit_transform(X[relevant_columns])
    XX = np.concatenate((XX, X ** 3), axis=1)
    return XX

XX_train = make_features(X_train, relevant)
XX_val = make_features(X_val, relevant)

pm = LinearRegression()
pm.fit(XX_train, y_train)
del XX_train # memory ...

predictions = pm.predict(XX_val)
mse_pm = mean_squared_error(y_val, predictions)
print(f"(Lasso Polynomial Model) MSE: {mse_pm}")