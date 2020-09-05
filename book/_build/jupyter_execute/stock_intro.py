# Data Description

In the following I will present my model for the stock data set. Before I could fit my final model I had to transform the data. Next I will discuss how I changed the data structure and how I chose split the data intro training and validation parts.

import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.getcwd()).parent

df = pd.read_parquet(ROOT / "data" / "stock_data.parquet")
df.iloc[:5, :10]

The original data set contains 1_629_155 observations of stock returns including 63 features. One of these features which is of particular importance is the date. In comparison to classical panel data, however, the above data does not have a unit index. That is, we cannot know which units move between time periods. Observations are measured from the 01.31.1965 until the 31.05.2014. Again, as in the simulated case, testing observations are marked with a ``NaN`` in the outcome column. Here the testing observations are all observations starting from the 31.01.2004 until the last observed time period. 

## Cleaning the Data

Before training my models I cleaned the data in several ways. First, I transformed the date column to a year column and a one-hot-encoded quarter column. I.e., ``data = 1965-01-31`` becomes ``year = 1965`` and all dummies will be zero, as the first quarter is integrated in the intercept. I then dropped all observations older than 1990. I did this since I believed that the any information in the data of the '70s-'90s which could be used to explain stock returns was unlikely to still explain modern stock returns. Also I wanted to reduce the size of the data set. At last I dropped all observations which had absolute returns greater than 6, as from looking at a fine histogram, these seemed to be outliers. The data cleaning script can be found here [clean_data.py](https://github.com/timmens/topics-project/blob/main/codes/clean_data.py).

## Train / Validation Split

As I did not want to ignore the time dimension for the train / validation split I constructed the respective sets as follows. I grouped the cleaned data set into sets by year. For each year I split the respective set into 80% training and 20% validation set. Lastly I concatened the smaller sets together to form the final training and validation sets. Using this strategy I can train my model on all time-periods and evaluate the performance on all time-periods. The specific implementation is given in the script [train_test_split.py](https://github.com/timmens/topics-project/blob/main/codes/train_test_split.py).