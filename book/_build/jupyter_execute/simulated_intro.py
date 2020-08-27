# Data Description

In the following I will present my model for the simulated data set. Before I present the final model I will talk about how I reached the decision and compared the model to other potential candidates.

Note first that I changed the file format from an ``RData`` file to a [``parquet``](https://parquet.apache.org/) file, as this can be loaded into most programming languages.

import os
from pathlib import Path
import pandas as pd
ROOT = Path(os.getcwd()).parent

df = pd.read_parquet(ROOT / "data" / "simulated_data.parquet")
df.iloc[:5, :10]

The data set consists of 100_000 observations of a single continuous outcome and 100 continuous features which have been transformed to a uniform distribution. Of the 100_000 observations 20_000 are designed for the testing step and are given a ``NaN`` in the outcome column. The remaining 80_000 *labelled* data points were (randomly) split further by me into 65_000 (81.25%) training points and 15_000 validation points. As is standard in the literature I will train all of my models on the training points and compare the performance on the validation points. The *best* model overall is trained on all 80_000 points and used to predict the outcomes on the test set.
This procedure is implemented in the script [train_test_split.py](https://github.com/timmens/topics-project/blob/main/train_test_split.py).

In the next section on "reverse engineering" I will present a few techniques I used to learn more about the data at hand. If you only care about the description of the models I considered then feel free to skip the next section.