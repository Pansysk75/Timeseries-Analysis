from utilities import *

# load dataset
dataset_all = read_datfile("./Datasets/eruption2001.dat")
dataset_small = trim_dataset(dataset_all, 500, alignment='start')

data = {"complete_dataset": dataset_all, "small_dataset": dataset_small}

# timeseries' plots
# for name, dataset in data.items():
#     plot_timeseries_stats(dataset, name, "./Part2/plots")

# first differences 
data_fd = {}
for name, dataset in data.items():
    data_fd[name] = np.diff(dataset, 1)

# firts differences plots
# for name, dataset in data_fd.items():
#     plot_timeseries_stats(dataset, f"{name}_first_diff", "./Part2/plots")

# delayed mutual information
# for name, dataset in data.items():
#     plot_delayed_mutual_information(dataset, 20, f"{name}")

tau = {"complete_dataset": 5, "small_dataset": 1}

# choice of embedding dimension m
# maxnum = {"complete_dataset": 320, "small_dataset": 40}
# for name, dataset in data.items():
#     # false nearest neighbors
#     m_max = 10
#     f1 = falsenearestneighbors(dataset, m_max=m_max, tau=tau[name], maxnum=maxnum[name], show=True, timeseries_name=name)

m = {"complete_dataset": 5, "small_dataset": 4}

# plot attractors
# for name, dataset in data.items():
#     embedded = embed_data(dataset, order=m[name], delay=tau[name])
#     # 3d attractor
#     #embedded = embed_data(dataset, order=3, delay=1)
#     plot_3d_attractor(embedded, name)

# correlation dimension
# for name, dataset in data.items():
#     correlationdimension(dataset, tau[name], m_max=10, show=True, timeseries_name=name)

# local predictions
for name, dataset in data.items():
    n_last = int(0.2*len(dataset))
    T_max = 3
    nrmse_llp, pred_llp = localpredictnrmse(dataset, n_last, m[name], tau[name], Tmax=T_max, nnei=20, q=m[name], timeseries_name=name)
    nrmse_lap, pred_lap = localpredictnrmse(dataset, n_last, m[name], tau[name], Tmax=T_max, nnei=20, q=0, timeseries_name=name)
    print(f"---{name}---")
    print(f"nrmse for 1 step llp: {nrmse_llp[0][0]}")
    print(f"nrmse for 1 step lap: {nrmse_lap[0][0]}")

    # plot 1 step predictions
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    # llp
    ax[0].plot(dataset[len(dataset) - n_last + T_max - 1: len(dataset)], label='true values')
    ax[0].plot(pred_llp[:, 0], label='predictions')

    ax[0].set_title("local linear model")
    ax[0].legend()

    # lap
    ax[1].plot(dataset[len(dataset) - n_last + T_max - 1: len(dataset)], label='true values')
    ax[1].plot(pred_lap[:, 0], label='predictions')

    ax[1].set_title("local average model")
    ax[1].legend()

    plt.savefig(f"./Part2/plots/local_pred_{name}.png")
    plt.show()
    plt.close()

  
    


