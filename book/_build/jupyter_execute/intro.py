import os

from pathlib import Path

from IPython.display import display
from IPython.display import FileLink

ROOT = Path(os.getcwd()).parent

# Introduction

***Topics in Econometrics and Statistics, University of Bonn, 2020 -- Prof. Joachim Freyberger***

In this document I present my submission to the project for the topics class in
econometrics and statistics, due by 13th September 2020.

## Problem Description

For the project we have been given a simulated data set and a real data set. For each
data set some of the observed outcomes are held back. The task is to build a model for
each data set on the training set and submit the predictions on the test set.


The following part of this document is structured as follows. At the end of this
section I present a link to my predictions for the respective data sets. In the next
chapter I present the model for the simulated data set. Aftwards I introduce my model
for the stock data set.

## Data Format

Please note that I changed the file format from an RData file to a [parquet](https://parquet.apache.org/) file, as this is not dependent on the specific programming language. If you want to reproduce my results using my code (see below) you need to transform the RData files to parquet files. This can easily be done, for example, with the [SparkR](https://spark.apache.org/docs/1.6.2/api/R/write.parquet.html) package for R. For the project to run I expect the two data files to be located in the folder ``data`` with names ``simulated_data.parquet`` and ``stock_data.parquet``, respectively.

## Code

In this project I have to sets of code bases. Nearly all of the code presented in the following notebooks is embedded in the notebooks themselves. For the final predictions however, I use Python scripts which are stored [here](https://github.com/timmens/topics-project/). The script [build_project.py](https://github.com/timmens/topics-project/tree/main/codes/build_project.py) runs all scripts that are necessary to produce the predictions. If you want to rerun these codes on your computer open your favorite terminal emulator and run the following line by line (I assume you have at least [miniconda](https://docs.conda.io/en/latest/miniconda.html) already installed on your system, otherwise read the note below.)

```console
$ git clone https://github.com/timmens/topics-project.git
$ conda env create -f environment.yml
$ conda activate topics-project
$ cd codes
$ python build_project.py
```

***Note.***

It is not necessary to use conda here as long as all the packages that I use are available to the Python interpreter. Conda just provides a very easy way to create a sandbox environment in which all packages will be available without messing with the system.

## Predictions

Predictions for the respective data sets are stored on github and can be downloaded here

- Simulated data: [(click here to download)](https://rawcdn.githack.com/timmens/topics-project/4045ae9fac293ec2aff62d4d9e6f3f8989f768cb/bld/predictions_simulated.csv)
- Stock data: [(click here to download)](https://rawcdn.githack.com/timmens/topics-project/4045ae9fac293ec2aff62d4d9e6f3f8989f768cb/bld/predictions_stock.csv)


```{toctree}
:hidden:
:titlesonly:


simulated
stock
```
