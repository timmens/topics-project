import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def correlation_heatmap(df=None, corr=None):
    if df is not None:
        corr = df.corr()

    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        vmin=0,
        vmax=1,
    )
