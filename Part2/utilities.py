from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from typing import Literal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from nolitsa import delay, dimension

def read_datfile(path):
    xV = np.loadtxt(path)
    return xV


def trim_dataset(xV, des_length, alignment: Literal['start', 'end', 'random']):
    '''Trim a dataset to des_length. 
    '''

    if alignment == 'start':
        trimmed_data = xV[0:des_length]
    elif alignment == 'end':
        trimmed_data = xV[-des_length:]
    elif alignment == 'random':
        offset = random.randint(0, len(xV) - des_length)
        trimmed_data = xV[offset:des_length + offset]
    else:
        raise ValueError('Unknown "alignment" argument')

    return trimmed_data


def split_dataset(xV, split_ratio = 0.8):
    split_idx = round(len(xV) * split_ratio)
    return (xV[0:split_idx], xV[split_idx+1:])

def plot_timeseries_stats(xV, name, savepath=None):
    '''Plot timeseries, auto-correlation, partial auto-correlation and portmanteau pvalues.'''
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(name, fontsize=16)

    plt.subplot(2, 1, 1).plot(xV)
    # plt.savefig("plots/timeseries/" + dataset_name + ".png")

    # From statsmodels module.
    plot_acf(xV, zero=False, lags=10, ax=plt.subplot(2,3,4))

    # From statsmodel module (yule-walker method).
    plot_pacf(xV, zero=False, lags=10, ax=plt.subplot(2,3,5), method='ywm')   

    # portmanteau test
    ljung_val, ljung_pval = acorr_ljungbox(xV, lags=20, return_df=False)
    
    ax = plt.subplot(2, 3, 6)
    ax.scatter(np.arange(1, len(ljung_pval) + 1), ljung_pval)
    ax.axhline(0.05, linestyle='--', color='r')
    ax.set_title('Ljung-Box Portmanteau test')
    ax.set_yticks(np.arange(0, 1.1))

    if(savepath != None):
        plt.savefig(savepath + "/" + name + ".png")

    plt.show()
    plt.close()

def plot_delayed_mutual_information(xV, maxtau, timeseries_name: str):
    mi = delay.dmi(xV, maxtau=maxtau + 1)[1:]

    fig, ax = plt.subplots()

    ax.plot(np.arange(1, maxtau + 1), mi, marker='o')
    ax.set(xlabel='lag τ', ylabel='Delayed mutual information')
    ax.set_title(timeseries_name)
    ax.set_xticks(np.arange(1, maxtau + 1))

    plt.savefig(f"./Part2/plots/mi_{timeseries_name}.png")
    plt.show()
    plt.close()

def falsenearestneighbors(xV, m_max=10, tau=1, maxnum=None, show=False, timeseries_name=None):
    dim = np.arange(1, m_max + 1)
    f1, _, _ = dimension.fnn(xV, tau=tau, dim=dim, window=10, maxnum=maxnum, metric='cityblock', parallel=False)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.scatter(dim, f1)
        ax.axhline(0.01, linestyle='--', color='red', label='1% threshold')
        ax.set_xlabel(f'm')
        ax.set_title(f'FNN {timeseries_name}')
        ax.set_xticks(dim)
        ax.legend()
        plt.savefig(f"./Part2/plots/fnn_{timeseries_name}.png")
        plt.show()
        plt.close()
    return f1

def plot_3d_attractor(xM, timeseries_name, connect_with_lines=False):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xM[:, 0], xM[:, 1], xM[:, 2])
    if connect_with_lines:
        ax.plot(xM[:, 0], xM[:, 1], xM[:, 2], linestyle='--')
    ax.set_title(f"3d attractor {timeseries_name}")

    plt.savefig(f"./Part2/plots/3d_{timeseries_name}")
    plt.show()
    plt.close()

def embed_data(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T