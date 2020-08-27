"""Split cleaned data sets into training, validation and testing sets.

"""
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).absolute().parent.parent


def _split_data(df, dataset):
    dfs = _split_stock_data(df) if dataset == "stock" else _split_simulated_data(df)
    return dfs


def _split_stock_data(df):
    fit, test = _split_into_fit_and_test(df)
    train, validate = _create_panel_hold_out(fit, time_column="year", train_size=0.8)
    return train, validate, fit, test


def _split_simulated_data(df):
    fit, test = _split_into_fit_and_test(df)
    train, validate = train_test_split(fit, train_size=0.8125, random_state=0)
    return train, validate, fit, test


def _split_into_fit_and_test(df):
    mask = df[["Y"]].isna().values
    fit = df.loc[~mask, :]
    test = df.loc[mask, :].drop("Y", axis=1).reset_index(drop=True)
    return fit, test


def _create_panel_hold_out(fit, time_column="year", train_size=0.8):
    """Create hold-out set for each time period and concatenate."""
    grouped = fit.groupby("year")

    train_data = []
    validate_data = []
    for year, group in grouped:
        train, validate = train_test_split(
            group, train_size=train_size, random_state=year
        )
        train_data.append(train)
        validate_data.append(validate)

    train = pd.concat(train_data, axis=0)
    validate = pd.concat(validate_data, axis=0)
    return train, validate


@click.command()
@click.option("--datasets", default=None)
def main(datasets):
    datasets = ["simulated", "stock"] if datasets is None else [datasets]
    for dataset in datasets:
        if dataset == "stock":
            fpath = ROOT / "bld" / "stock_data_clean.parquet"
        else:
            fpath = ROOT / "data" / "simulated_data.parquet"

        df = pd.read_parquet(fpath)
        train, validate, fit, test = _split_data(df, dataset)

        test.to_parquet(ROOT / "bld" / f"test_{dataset}.parquet")
        train.to_parquet(ROOT / "bld" / f"train_{dataset}.parquet")
        validate.to_parquet(ROOT / "bld" / f"validate_{dataset}.parquet")
        fit.to_parquet(ROOT / "bld" / f"fit_{dataset}.parquet")


if __name__ == "__main__":
    main()
