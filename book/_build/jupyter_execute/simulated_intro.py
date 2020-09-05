# Data Description

Let us first look at a few observations of the data set.

import os
from pathlib import Path
import pandas as pd
ROOT = Path(os.getcwd()).parent

df = pd.read_parquet(ROOT / "data" / "simulated_data.parquet")
df.iloc[:5, :10]

The data set consists of 100_000 observations of a single continuous outcome and 100 continuous features which have been transformed to a uniform distribution. Of the 100_000 observations 20_000 are designed for the testing step and are marked by a ``NaN`` in the outcome column.

## Train / Validation Split

I (randomly) split the remaining 80_000 *labelled* data points into 65_000 (81.25%) training points and 15_000 validation points. As is standard in the literature I will train all of my models on the training points and compare the performance on the validation points. The *best* model overall is then trained on all 80_000 points and used to predict the outcomes on the test set.
This splitting procedure is implemented in the script [train_test_split.py](https://github.com/timmens/topics-project/blob/main/codes/train_test_split.py).

## Next Up

In the next section on "reverse engineering" I will present a few techniques I used to learn more about the data at hand. If you only care about the final model I considered then feel free to skip this section.