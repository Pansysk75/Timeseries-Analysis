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


Trim datasets to size of smallest dataset
data = trim_datasets(data, alignment="start")



Path("plots").mkdir(exist_ok=True)

# # Plot stats for non-stationary timeseries
# for dataset_name, dataset in data.items():
#     plot_timeseries_stats(xV = dataset,
#                           name = dataset_name,
#                           savepath = "plots")

# Take first difference and plot again
data_fd = {}
for dataset_name, dataset in data.items():
    data_fd[dataset_name] = first_diff(dataset)
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
    train_set[name], test_set[name] = split_dataset(xV, split_ratio=0.8)


# Get train/test sets and take first difference
name = "eruption2000.dat"

xV = train_set[name]
xV_test = test_set[name]

# xV = first_diff(train_set[name])
# xV_test = first_diff(test_set[name])

# Fit ARMA models
best_model, _ = batch_arima_test(xV, p_min=0, p_max=5,
                     q_min=0, q_max=2, d=1,
                     show=True)


# Get predictions for train set
predM = in_sample_predict_ahead(xV, best_model, 2)

# Get predictions for test set
best_model = best_model.apply(xV_test) 
predM_test = in_sample_predict_ahead(xV_test, best_model, 2)

# # Reverse the first-difference process for true data
# xV = reverse_first_diff(train_set[name][0], xV)[1:]
# xV_test = reverse_first_diff(test_set[name][0], xV_test)[1:]

# # Go from first-difference to true-scale values for generated data
# predM = [np.add(train_set[name][:-1], ts) for ts in predM]
# predM_test = [np.add(test_set[name][:-1], ts) for ts in predM_test]


# Plot everything!
fig = plt.figure(figsize=(12,8))
fig.suptitle(name, fontsize=16)

plt.subplot(3,1,2)
plt.plot(xV)
for _xV in predM:
    plt.plot(_xV, linestyle='dashed')
plt.title(f'Train set: NRMSE={nrmse(xV, predM[0]):.4f}')


plt.subplot(3,2,5)
plt.plot(xV_test)
for _xV in predM_test:
    plt.plot(_xV, linestyle='dashed')
plt.title(f'Test set: NRMSE={nrmse(xV_test, predM_test[0]):.4f}')

plt.subplot(3,2,6)
resid = xV - predM[0]
plt.scatter(np.arange(len(resid)), resid)
plt.title("Residuals")
# Make y axis limits symmetric
y_bound = np.max(np.abs(plt.ylim()))
plt.ylim(-y_bound, y_bound)

plt.show()

