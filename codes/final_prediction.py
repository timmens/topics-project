"""Script to create the final predictions using catboost and lasso / least-squares.

In this script I fit the final model to the complete training set and predict the
outcomes of the test set.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).absolute().parent.parent


def fit(X_train, y_train, dataset, **kwargs):
    """Fit a catboost model on training data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        dataset (str): Dataset type. Must be in {'simulated', 'stock'}.
        kwargs (dict): Keyword arguments for respective fitting procedures.

    Returns:
        regressor (catboost.CatboostRegressor or sklearn.linear_model.LinearRegression):
            The fitted regressor.

    """
    if dataset == "simulated":
        regressor = simulated_model(X_train, y_train, **kwargs)
    elif dataset == "stock":
        regressor = stock_model(X_train, y_train, **kwargs)
    else:
        raise NotImplementedError
    return regressor


def simulated_model(
    X_train,
    y_train,
    iterations=1500,
    learning_rate=0.01,
    depth=5,
    loss_function="RMSE",
    random_state=1,
):
    """Final model for the simulated data set.

    Here I fit gradient boosted trees using the catboost library. For details please
    seek the theory part of my project.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training labels.
        iterations (int): Number of trees to use in model.
        learning_rate (float): Learning rate in model.
        depth (int): Depth of oblivious trees.
        loss_function (callable): Loss function which is used for calibration.
        random_state (int): Seed for random number generator.

    Returns:
        regressor (catboost.CatBoostRegressor): Fitted catboost regressor.

    """
    regressor = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        random_state=random_state,
    )

    regressor.fit(X_train, y_train, verbose=100)
    return regressor


def stock_model(X_train, y_train, alphas=None, cv=5, random_state=1):
    """Final model for the stock data set.

    Algorithm:
        1. Fit Lasso using cross validation and select non-zero coefs columns
        2. Create 2nd degree polynomial features including (without interaction)
            the third power.
        3. On this set of features fit a linear regression model

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training labels.
        alphas (np.ndarray): 1d array of regularization parameters to go through
            during cross validation.
        cv (int): Number of cross validation folds.

    Returns:
        out (dict): Dictionary containing 'regressor' the fitted model
            (sklearn.linear_model.LinearRegression) and 'relevant' a list of relevant
            feature names.

    """
    alphas = np.logspace(-2.5, 1, 50) if alphas is None else alphas

    lasso_model = LassoCV(alphas=alphas, cv=cv, random_state=random_state)
    lasso_model.fit(X_train, y_train)
    relevant = X_train.columns[lasso_model.coef_ != 0].to_list()

    XX_train = transform_features_stock(X_train, relevant)

    regressor = LinearRegression().fit(XX_train, y_train)
    out = {"regressor": regressor, "relevant": relevant}
    return out


def transform_features_stock(X, relevant):
    """Transform features for stock data set."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = X[relevant]
    XX = poly.fit_transform(X)
    XX = np.concatenate((XX, X ** 3), axis=1)
    return XX


def predict(regressor, X_test):
    """Predict on new data given fitted regressor.

    Args:
        regressor (catboost.CatboostRegressor or dict): The fitted regressor.
        X_test (pd.DataFrame): Test data of shape equivalent as used to fit `regressor`

    Returns:
        prediction (np.ndarray): The predictions.

    """
    if isinstance(regressor, dict):
        relevant = regressor["relevant"]
        regressor = regressor["regressor"]
        X_test = transform_features_stock(X_test, relevant)

    prediction = regressor.predict(X_test)
    return prediction


def fit_kwargs(dataset):
    """Return fit kwargs given dataset.

    Args:
        dataset (str): Type of data set, must be in {'simulated', 'stock'}.

    Returns:
        kwargs (dict): Keyword arguments for function `fit`.

    """
    if dataset == "simulated":
        kwargs = {
            "iterations": 1500,
            "learning_rate": 0.01,
            "depth": 5,
            "loss_function": "RMSE",
            "random_state": 1,
        }
    elif dataset == "stock":
        kwargs = {
            "alphas": np.logspace(-2.5, 1, 50),
            "cv": 5,
            "random_state": 1,
        }
    else:
        raise NotImplementedError
    return kwargs


def main():
    for dataset in ["simulated", "stock"]:
        df_fit = pd.read_parquet(ROOT / "bld" / f"fit_{dataset}.parquet")
        X_test = pd.read_parquet(ROOT / "bld" / f"test_{dataset}.parquet")

        y_fit = df_fit["Y"]
        X_fit = df_fit.drop("Y", axis=1)

        kwargs = fit_kwargs(dataset)
        regressor = fit(X_fit, y_fit, dataset, **kwargs)
        predictions = predict(regressor, X_test)
        np.savetxt(
            ROOT / "bld" / f"predictions_{dataset}.csv", predictions, delimiter=","
        )


if __name__ == "__main__":
    main()
