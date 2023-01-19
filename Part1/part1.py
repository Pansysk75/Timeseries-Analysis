import numpy as np
from utilities import *
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### Import datasets
dataset_path = "/home/panos/Desktop/timeseries/Datasets/"
dataset_files = ["eruption1989.dat","eruption2000.dat", "eruption2011.dat"]

data = {}

# ts = []
# p = (0.0, 0.5, 0.1, -0.2)
# ts.extend(np.random.standard_normal(len(p)))
# for i in range(1000):
#     l = len(ts)
#     next = p[0]
#     next += sum([ts_*p_ for (ts_,p_) in zip(reversed(ts), p[1:])])
#     next += np.random.standard_normal(1)[0]
#     ts.append(next)
# # print(ts)
# # data["synthetic"] = ts

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

### ARIMA part:
for name, ts in data.items():
    ts = ts[:100]
    split_idx = round(0.8*len(ts))
    xV = ts[:split_idx]
    xV_test = ts[split_idx+1:]
    ### Fit many ARIMA models (first difference) and choose best
    ### depending on aic value
    d=1
    # best_model, aic_values = batch_arima_test(xV, p_min=0, p_max=6,
    #                     q_min=0, q_max=6, d=d,
    #                     show=True, name=name)
    # plot_aic_grid(aic_values, name, "plots")
    p_values = np.arange(1, 51)
    error = []
    aic = []
    bic = []
    fpe = []
    for order in [(i,1,0) for i in p_values]:
        # best_model = sm.tsa.SARIMAX(xV, order=order, enforce_invertibility=False, enforce_stationarity=False).fit(maxiter=1000)
        best_model = sm.tsa.SARIMAX(xV, order=order).fit()


        ### Get in-sample predictions (T indicates how many time-steps
        ### ahead to predict)
        T_list = [1, 2]
        predM = in_sample_predict_ahead(xV, best_model, T_list)


        ### Start plotting fit!
        fig = plt.figure(figsize=(5,2))
        # fig.suptitle(name, fontsize=16)

        ### Plot true & 1-ahead prediction
        plt.subplot(1,1,1)
        plt.plot(xV)
        plt.plot(predM[0], linestyle='dashed',linewidth=0.9)
        p = len(best_model.arparams)
        q = len(best_model.maparams)
        err = nrmse(xV, predM[0])
        res = xV - predM[0]
        plt.title(f"({p},{d},{q}) | NRMSE={err:.4f}")

        ### Save  plot
        plt.tight_layout()
        # plt.savefig("plots/"+name+"_predictions_" + str(order) + ".png", dpi=200)
        plt.show()
        


        # Predict on test data
        pred_new_data =predict_ahead(xV_test, best_model, [1])[0]
          ### Start plotting fit!
        fig = plt.figure(figsize=(4,2))
        # fig.suptitle(name, fontsize=16)

        ### Plot true & 1-ahead prediction
        plt.subplot(1,1,1)
        plt.plot(xV_test)
        plt.plot(pred_new_data, linestyle='dashed',linewidth=0.9)
        p = len(best_model.arparams)
        q = len(best_model.maparams)
        err = nrmse(xV_test, pred_new_data)
        plt.title(f"({p},{d},{q}) | NRMSE={err:.4f}")

        ### Save  plot
        plt.tight_layout()
        # plt.savefig("plots/"+name+"_predictions_" + str(order) + "_n.png", dpi=200)
        plt.show()



        ### Save aic, bic, fpe data
        var_resid = np.nanvar(res)
        p = order[0]
        n = len(xV)
        error.append(var_resid)
        aic.append(np.log(var_resid) + 2*p/n)
        bic.append(np.log(var_resid) + 2*p*np.log(n)/n)
        fpe.append(var_resid * (n+p)/(n-p))

    ### Plot aic, bic, fpe
    fig = plt.figure(figsize=(8,3))
    plt.plot(p_values, error, label="error")
    plt.plot(p_values, aic, label="aic")
    plt.plot(p_values, bic, label="bic")
    plt.plot(p_values, fpe, label="fpe")
    plt.legend()
    plt.savefig("plots/"+name+"_ic.png", dpi=200)

    np.savetxt(name + "_ic_smalln.txt", (p_values, error, aic, bic, fpe))
    exit()


