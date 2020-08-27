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

## Code

In this project I have to sets of code bases. Nearly all of the code presented in the following notebooks is embedded in the notebooks itself. For the final predictions however, I use Python scripts which are stored [here](https://github.com/timmens/topics-project/tree/main/codes). The script [create_predictions.py](https://github.com/timmens/topics-project/tree/main/codes/main.py) runs all scripts that are necessary to produce the predictions. If you want to rerun these codes on your computer open your favorite terminal emulator and run the following line by line (I assume you have at least [miniconda](https://docs.conda.io/en/latest/miniconda.html) already installed on your system, otherwise read the note below.)

```zsh
git clone https://github.com/timmens/topics-project.git
conda env create -f environment.yml
conda activate topics-project
cd codes
python create_predictions.py
```

***Note.***

It is not necessary to use conda here as long as all the packages that I use are available to the Python interpreter. Conda just provides a very easy way to create a sandbox environment in which all packages will be available without messing with the system.

## Predictions

file_simulated = FileLink(
    ROOT / "bld" / "simulated.csv",
    result_html_prefix="Click here to download predictions for the simulated data: "
)
file_stock = FileLink(
    ROOT / "bld" / "stock.csv",
    result_html_prefix="Click here to download predictions for the stock data: "
)

display(file_simulated)

display(file_stock)

- Simulated data: [(click here to download predictions)](https://github.com/timmens/topics-project)
- Stock data: [(click here to download predictions)](https://github.com/timmens/topics-project)


```{toctree}
:hidden:
:titlesonly:


simulated
content
```
