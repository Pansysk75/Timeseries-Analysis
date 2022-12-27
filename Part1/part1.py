import pandas as pd
import numpy as np

from utilities import *

from pathlib import Path


# Import datasets
dataset_path = "../Datasets/"
dataset_files = ["eruption1989.dat", "eruption2000.dat", "eruption2011.dat"]

data = {}
for dataset in dataset_files:
    data[dataset] = read_datfile(dataset_path + dataset)


# Trim datasets to size of smallest dataset
data = trim_datasets(data, alignment="random")

Path("plots").mkdir(exist_ok=True)

# Plot stats for non-stationary timeseries
for dataset_name, dataset in data.items():
    plot_timeseries_stats(dataset, dataset_name, "plots")

# Take first difference and plot again
for dataset_name, dataset in data.items():
    dataset = np.diff(dataset, 3)
    plot_timeseries_stats(dataset, dataset_name+"(1st_diff)", "plots")