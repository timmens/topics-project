# Data Description

In the following I will present my model for the stock data set. Before I could fit my final model I had to transform the data. Next I will discuss how I changed the data structure and how I split the data intro training and validation parts.

import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.getcwd()).parent

df = pd.read_parquet(ROOT / "data" / "stock_data.parquet")
df.iloc[:5, :10]

The original data set contains 1_629_155 observations of stock returns including 63 features. One of these features is the date. In comparison to classical panel data, however, the above data does not have a unit index. That is, we cannot know which units move between time periods. Observations are measured from the 01.31.1965 until the 31.05.2014. Again, as in the simulated case, testing observations are given a ``NaN`` in the outcome column. Here the testing observations are all observations starting from the 31.01.2004 until the last observed time period. 

## Cleaning the Data

Before training my models on the data I cleaned it in several ways. First of all I transformed the date column to a year column. I then dropped all observations older than 1990. And at last I dropped all observations which had absolute stock returns greater than 6. The data cleaning script can be found here [clean_data.py](https://github.com/timmens/topics-project/blob/main/codes/clean_data.py).

## Train / Validation Split

As I did not want to ignore the time dimension for the train / validation split I constructed the respective sets as follows. I grouped the cleaned data set into smaller sets by year. For each year I split the smaller set into 80% training and 20% validation set. Lastly I concatened the smaller sets together to form the final training and validation sets. Using this strategy I can train my model on all time-periods and evaluate the performance on all time-periods. The specific implementation is given in the script [train_test_split.py](https://github.com/timmens/topics-project/blob/main/codes/train_test_split.py).