"""This script creates final predictions for both data sets.

This script assumes that the data has been formatted from RData to parquet and is stored
with the correct names `simulated_data.parquet` and `stock_data.parquet`.
"""
import subprocess


def main():
    tasks = [
        "clean_data.py",
        "train_test_split.py",
        "regplots.py",
        "final_prediction.py",
    ]
    for task in tasks:
        subprocess.run(["python", task])


if __name__ == "__main__":
    main()
