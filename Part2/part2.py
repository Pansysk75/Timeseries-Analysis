from utilities import *
import matplotlib
matplotlib.use('tkagg')


# load dataset
dataset_all = read_datfile("./Datasets/eruption2001.dat")
dataset_small = trim_dataset(dataset_all, 500, alignment='start')

data = {"complete_dataset": dataset_all, "small_dataset": dataset_small}

# timeseries' plots
for name, dataset in data.items():
    plot_timeseries_stats(dataset, name, "./Part2/plots")

# # first differences 
# data_fd = {}
# for name, dataset in data.items():
#     data_fd[name] = np.diff(dataset, 1)

# # first differences plots
# for name, dataset in data_fd.items():
#     plot_timeseries_stats(dataset, f"{name}_first_diff", "./Part2/plots")

# # delayed mutual information
# for name, dataset in data.items():
#     plot_delayed_mutual_information(dataset, 20, f"{name}")

tau = {"complete_dataset": 5, "small_dataset": 1}

# # choice of embedding dimension m
# maxnum = {"complete_dataset": 320, "small_dataset": 40}
# for name, dataset in data.items():
#     # false nearest neighbors
#     m_max = 10
#     f1 = falsenearestneighbors(dataset, m_max=m_max, tau=tau[name], maxnum=maxnum[name], show=True, timeseries_name=name)

m = {"complete_dataset": 5, "small_dataset": 4}

# # plot attractors
# for name, dataset in data.items():
#     embedded = embed_data(dataset, order=m[name], delay=tau[name])
#     # 3d attractor
#     #embedded = embed_data(dataset, order=3, delay=1)
#     plot_3d_attractor(embedded, name)

# correlation dimension
for name, dataset in data.items():

    # This doesn't work well for large m
    plot_correlation_dimension(dataset, tau[name], m_max=3, name = name)

    # This works fine but assumes tau=1
    plot_correlation_dimension_2(dataset, m_max=10, rmin=1.5, rmax=50, name = name)

    correlationdimension(dataset, m_max=10, show=True, name=name)

# local predictions
for name, dataset in data.items():
    n_last = int(0.2*len(dataset))
    T_max = 3
    nrmse_llp, pred_llp = localpredictnrmse(dataset, n_last, m[name], tau[name], Tmax=T_max, nnei=20, q=m[name], timeseries_name=name)
    nrmse_lap, pred_lap = localpredictnrmse(dataset, n_last, m[name], tau[name], Tmax=T_max, nnei=20, q=0, timeseries_name=name)

    print(f"{name} nrmse llp: {nrmse_llp}")
    print(f"{name} nrmse lap: {nrmse_lap}")

    # plot 1 step predictions
    
    ### llp
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(name, fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(dataset[len(dataset) - n_last + T_max - 1: len(dataset)], label='true values')
    plt.plot(pred_llp[:, 0], label='predictions')

    nrmse_string = "{:.4f}".format(nrmse_llp[0][0])
    plt.title(f"local linear model NRMSE={nrmse_string}")
    plt.legend()

    # Plot residuals
    plt.subplot(3, 1, 2)
    resid = dataset[len(dataset) - n_last + T_max - 1: len(dataset)] - pred_llp[:, 0]
    resid = resid[~np.isnan(resid)]
    plt.scatter(np.arange(len(resid)), resid)
    plt.title("Residuals")
    ### (Make y axis limits symmetric)
    y_bound = np.max(np.abs(plt.ylim()))
    plt.ylim(-y_bound, y_bound)

    # Plot residuals autocorrelation
    ax = plt.subplot(3,2,5)
    plot_pacf(resid, zero=False, lags=10, ax=ax, method='ywm')
    plt.title("Residuals PACF")

    # Plot residuals partial autocorrelation
    ax = plt.subplot(3,2,6)
    plot_acf(resid, zero=False, lags=10, ax=ax)
    plt.title("Residuals ACF")

    plt.savefig(f"./Part2/plots/llp_{name}.png")
    plt.show()
    plt.close()


    ### lap
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(name, fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(dataset[len(dataset) - n_last + T_max - 1: len(dataset)], label='true values')
    plt.plot(pred_lap[:, 0], label='predictions')

    nrmse_string = "{:.4f}".format(nrmse_lap[0][0])
    plt.title(f"local average model NRMSE={nrmse_string}")
    plt.legend()

    # Plot residuals
    plt.subplot(3, 1, 2)
    resid = dataset[len(dataset) - n_last + T_max - 1: len(dataset)] - pred_lap[:, 0]
    resid = resid[~np.isnan(resid)]
    plt.scatter(np.arange(len(resid)), resid)
    plt.title("Residuals")
    ### (Make y axis limits symmetric)
    y_bound = np.max(np.abs(plt.ylim()))
    plt.ylim(-y_bound, y_bound)

    # Plot residuals autocorrelation
    ax = plt.subplot(3,2,5)
    plot_pacf(resid, zero=False, lags=10, ax=ax, method='ywm')
    plt.title("Residuals PACF")

    # Plot residuals partial autocorrelation
    ax = plt.subplot(3,2,6)
    plot_acf(resid, zero=False, lags=10, ax=ax)
    plt.title("Residuals ACF")

    plt.savefig(f"./Part2/plots/lap_{name}.png")
    plt.show()
    plt.close()
    

# test linear model from 1st part
for name, dataset in data.items():
    n = len(dataset)
    train, test = dataset[:int(0.8 * n)], dataset[int(0.8*n):n]

    model = ARIMA(train, order=(1, 1, 1)).fit()
    pred = ARIMA_out_of_sample_predict_ahead(test, model)

    # plot
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(name, fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(int(0.8 * n), n), test, label='true values')
    plt.plot(np.arange(int(0.8 * n), n), pred, label='predictions')
    nrmse_string = "{:.4f}".format(nrmse(test, pred))
    plt.title(f"ARIMA(1,1,1), NRMSE={nrmse_string}")
    plt.legend()

    # Plot residuals
    plt.subplot(3, 1, 2)
    resid = test - pred
    resid = resid[~np.isnan(resid)]
    plt.scatter(np.arange(len(resid)), resid)
    plt.title("Residuals")
    ### (Make y axis limits symmetric)
    y_bound = np.max(np.abs(plt.ylim()))
    plt.ylim(-y_bound, y_bound)

    # Plot residuals autocorrelation
    ax = plt.subplot(3,2,5)
    plot_pacf(resid, zero=False, lags=10, ax=ax, method='ywm')
    plt.title("Residuals PACF")

    # Plot residuals partial autocorrelation
    ax = plt.subplot(3,2,6)
    plot_acf(resid, zero=False, lags=10, ax=ax)
    plt.title("Residuals ACF")

    plt.tight_layout()
    plt.savefig(f"Part2/plots/arima_test_{name}.png")
    plt.show()
    plt.close()