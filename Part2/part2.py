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

# # split train test
# train_data = {}
# test_data = {}
# for name, dataset in data.items():
#     train, test = split_dataset(dataset, 0.8)
#     train_data[name] = train
#     test_data[name] = test

# delayed mutual information
# for name, dataset in data.items():
#     plot_delayed_mutual_information(dataset, 20, f"{name}")

tau = {"complete_dataset": 5, "small_dataset": 3}

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
for name, dataset in data.items():
    correlationdimension(dataset, tau[name], m_max=10, show=True, timeseries_name=name)
    


