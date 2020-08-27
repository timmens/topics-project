"""Script to make regression plots.

This script is supposed to be called from the command line.

Example:
--------
"Plotting only first order terms (no-interaction) and for each (Y, Xi) plot fit a 3
degree polynomial model to the data":

    python regplots.py -order 3 --no-interaction


Arguments:
----------

--no-interaction (flag): If set then only first order terms are plotted
--overwrite (flag): If set then all existing data is overwritten
-order (int): Positive integer representing the order of polynomial fitted to each plot

"""
from functools import partial
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from joblib import delayed
from joblib import Parallel
from joblib import parallel_backend
from sklearn.preprocessing import PolynomialFeatures


ROOT = Path(__file__).absolute().parent.parent


def _plot(col_name, df_path, plot_path, order):
    """Make sns.lmplot and save."""
    df = pq.read_table(df_path, columns=["Y", col_name]).to_pandas()

    plot = sns.lmplot(
        col_name,
        "Y",
        data=df,
        scatter_kws={"alpha": 0.1},
        line_kws={"color": "black"},
        fit_reg=True,
        order=order,
        ci=None,
    )
    plot.set(ylim=(-15, 15))
    plot.set(xlim=(0, 1))

    plot_path = plot_path / f"{col_name}.png"
    plot.savefig(plot_path)
    plt.close(plot.fig)


def _make_no_interaction_plots(df, order):
    X = df.drop("Y", axis=1)

    kwargs = {
        "df_path": ROOT / "bld" / "train_simulated.parquet",
        "plot_path": ROOT / "bld" / "figures" / "no_interaction",
        "order": order,
    }
    func = partial(_plot, **kwargs)

    columns = X.columns
    with parallel_backend("multiprocessing", n_jobs=4):
        Parallel()(delayed(func)(col) for col in columns)


def _save_polynomial_features(df, poly_path, overwrite=True):
    if not poly_path.is_file() or overwrite:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

        X = df.drop("Y", axis=1)
        XX = poly.fit_transform(X)
        XX = pd.DataFrame(XX, columns=poly.get_feature_names(X.columns))
        XX.columns = XX.columns.str.replace(" ", "times")

        # save data since it is too big to be in memory
        df = pd.concat([df[["Y"]], XX], axis=1)
        df.to_parquet(poly_path)


def _make_interaction_plots(df, poly_path, order):
    kwargs = {
        "df_path": poly_path,
        "plot_path": ROOT / "bld" / "interaction",
        "order": order,
    }
    func = partial(_plot, **kwargs)

    columns = pq.read_schema(poly_path).names
    columns = [col for col in columns if col.startswith("X")]

    with parallel_backend("multiprocessing", n_jobs=4):
        Parallel()(delayed(func)(col) for col in columns)


@click.command()
@click.option("--no-interaction", is_flag=True, default=True)
@click.option("--overwrite", is_flag=True, default=True)
@click.argument("order", type=int, default=3)
def main(no_interaction, overwrite, order):
    (ROOT / "bld" / "figures" / "no_interaction").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "bld" / "train_simulated.parquet")
    _make_no_interaction_plots(df, order)
    if not no_interaction:
        poly_path = ROOT / "bld" / "polynomial_simulated.parquet"
        _save_polynomial_features(df, poly_path, overwrite)
        _make_interaction_plots(df, poly_path, order)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
