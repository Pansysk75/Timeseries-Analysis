import numpy as np
from utilities import *
from pathlib import Path
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

### Import datasets
dataset_path = "/home/panos/Desktop/timeseries/Datasets/"
dataset_files = ["eruption1989.dat", "eruption2000.dat", "eruption2011.dat"]

data = {}
for dataset in dataset_files:
    data[dataset] = read_datfile(dataset_path + dataset)


### Trim datasets to size of smallest dataset
data = trim_datasets(data, alignment="start")



Path("plots").mkdir(exist_ok=True)

### Plot stats for non-stationary timeseries
for dataset_name, dataset in data.items():
    plot_timeseries_stats(xV = dataset,
                          name = dataset_name,
                          savepath = "plots")

### Take first difference and plot again
data_fd = {}
for dataset_name, dataset in data.items():
    data_fd[dataset_name] = first_diff(dataset)
    plot_timeseries_stats(xV = data_fd[dataset_name],
                          name = dataset_name+"(1st_diff)",
                          savepath = "plots")


### ARIMA part:
for name, xV in data.items():

    ### Fit many ARIMA models (first difference) and choose best
    ### depending on aic value
    d=1
    best_model, aic_values = batch_arima_test(xV, p_min=0, p_max=6,
                        q_min=0, q_max=6, d=d,
                        show=True, name=name)
    plot_aic_grid(aic_values, name, "plots")

    ### Get in-sample predictions (T indicates how many time-steps
    ### ahead to predict)
    T_list = [1, 2, 3, 4, 5]
    predM = in_sample_predict_ahead(xV, best_model, T_list)


    ### Start plotting summary!
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(name, fontsize=16)

    ### Plot true & 1-ahead prediction
    plt.subplot(3,1,1)
    plt.plot(xV)
    plt.plot(predM[0], linestyle='dashed',linewidth=0.9)
    p = len(best_model.arparams)
    q = len(best_model.maparams)
    err = nrmse(xV, predM[0])
    plt.title(f"Fitted ARIMA model: ({p},{d},{q})\nNRMSE={err:.4f}")

    ### Plot true & more that 1-ahead predictions
    plt.subplot(3,1,2)
    plt.plot(xV)
    for _xV in predM[1:]:
        plt.plot(_xV, linestyle='dashed',linewidth=0.9)
    T_string = ", ".join(map(str, T_list[1:]))
    plt.title(f"Predict ahead for {T_string} time steps")


    ### Plot residuals
    plt.subplot(3,3,7)
    resid = xV - predM[0]
    resid = resid[~np.isnan(resid)]
    plt.scatter(np.arange(len(resid)), resid)
    plt.title("Residuals")
    ### (Make y axis limits symmetric)
    y_bound = np.max(np.abs(plt.ylim()))
    plt.ylim(-y_bound, y_bound)

    ### Plot residuals autocorrelation
    ax = plt.subplot(3,3,8)
    plot_pacf(resid, zero=False, lags=10, ax=ax, method='ywm')
    plt.title("Residuals PACF")

    ### Plot residuals partial autocorrelation
    ax = plt.subplot(3,3,9)
    plot_acf(resid, zero=False, lags=10, ax=ax)
    plt.title("Residuals ACF")

    ### Save summary plot
    plt.tight_layout()
    plt.savefig("plots/"+name+"_predictions.png")
    plt.show()

