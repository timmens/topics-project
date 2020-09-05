"""Script to clean data -- is only used for stock data.

To simplify the problem, with the hope of not loosing too much precision, we
transform the daily dates to yearly dates; we drop observations older than 1990
and we winsorize the data by dropping all observations with an absolute outcome
higher than 6.

"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).absolute().parent.parent


def _transform_date_to_year_and_quarter_column(df):
    date = pd.to_datetime(df["date"])
    df["year"] = date.dt.year

    quarters = pd.get_dummies(date.dt.quarter).iloc[:, 1:]
    quarters.columns=["quarter2", "quarter3", "quarter4"]

    df = df.drop("date", axis=1)
    df = pd.concat((df, quarters), axis=1)
    return df


def _drop_old_observations(df, oldest_year):
    df = df.query("year > @oldest_year")
    return df


def _drop_outliers(df, threshold):
    df = df.query("Y.abs() < @threshold or Y.isnull()")
    return df


def main():
    df = pd.read_parquet(ROOT / "data" / "stock_data.parquet")

    df = _transform_date_to_year_and_quarter_column(df)
    df = _drop_old_observations(df, oldest_year=1990)
    df = _drop_outliers(df, threshold=6)
    df = df.reset_index(drop=True)

    df.to_parquet(ROOT / "bld" / "stock_data_clean.parquet")


if __name__ == "__main__":
    main()
