import pandas as pd
import numpy as np

from utilities import *

import pmdarima as pmd

from pathlib import Path

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import datasets
dataset_path = "../Datasets/"
dataset_files = ["eruption1989.dat", "eruption2000.dat", "eruption2011.dat"]

data = {}
for dataset in dataset_files:
    data[dataset] = read_datfile(dataset_path + dataset)


# Trim datasets to size of smallest dataset
data = trim_datasets(data, alignment="random")



Path("plots").mkdir(exist_ok=True)

# # Plot stats for non-stationary timeseries
# for dataset_name, dataset in data.items():
#     plot_timeseries_stats(xV = dataset,
#                           name = dataset_name,
#                           savepath = "plots")

# Take first difference and plot again
data_fd = {}
for dataset_name, dataset in data.items():
    data_fd[dataset_name] = np.diff(dataset, 1)
    # plot_timeseries_stats(xV = data_fd[dataset_name],
    #                       name = dataset_name+"(1st_diff)",
    #                       savepath = "plots")

# # Take log + first difference and plot again
# data_fd_log = {}
# for dataset_name, dataset in data.items():
#     data_fd_log[dataset_name] = np.diff(np.log(dataset), 1)
#     plot_timeseries_stats(data_fd_log[dataset_name], dataset_name+"(1st_diff_log)", "plots")

# Split into train and test datasets
train_set = {}
test_set = {}

for name, xV in data.items():
    train_set[name], test_set[name] = split_dataset(xV)

# Fit AR models
xV = train_set["eruption1989.dat"]
best_model, _ = batch_arima_test(xV, p_min=0, p_max=3,
                     q_min=0, q_max=3,
                     show=True)


best_model.plot_diagnostics()
print("Summary for trained model:")
print(best_model.summary())
xV_pred = best_model.predict()

xV_test = test_set["eruption1989.dat"]
res2 = best_model.apply(xV_test)
print("Summary for test data:")
print(res2.summary())
xV_test_pred = res2.predict()
res2.plot_diagnostics()


plt.figure()
plt.plot(xV)
plt.plot(xV_pred)
plt.title(f'NRMSE: {nrmse(xV, xV_pred)}')
plt.show()

plt.figure()
plt.plot(xV_test)
plt.plot(xV_test_pred)
plt.title(f'NRMSE: {nrmse(xV, xV_pred)}')
plt.show()

