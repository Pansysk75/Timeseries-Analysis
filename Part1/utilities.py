from typing import Literal
import random

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

###!pip install numpy matplotlib statsmodels


def read_datfile(path):
    xV = np.loadtxt(path)
    return xV


def trim_datasets(data, alignment: Literal['start', 'end', 'random']):
    '''Trim all datasets to the size of the smallest dataset. For longer datasets,
    we select a random portion the size of the smallest dataset.
    '''
    trimmed_data = {}

    data_length = min(len(v) for v in data.values())

    if alignment == 'start':
        trimmed_data = {key: value[0:data_length] for (key, value) in data}
    elif alignment == 'end':
        trimmed_data = {key: value[-data_length:] for (key, value) in data}
    elif alignment == 'random':
        for dataset in data:
            offset = random.randint(0, len(data[dataset]) - data_length)
            trimmed_data[dataset] = data[dataset][offset:data_length + offset]
    else:
        raise ValueError('Unknown "alignment" argument')

    return trimmed_data


def plot_timeseries_stats(xV, name, savepath=None):
    '''Plot timeseries, auto-correlation and partial auto-correlation.'''
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(name, fontsize=16)

    plt.subplot(2, 1, 1).plot(xV)
    # plt.savefig("plots/timeseries/" + dataset_name + ".png")

    plot_acf(xV, zero=False, lags=10, ax=plt.subplot(2,2,3))
    plot_pacf(xV, zero=False, lags=10, ax=plt.subplot(2,2,4), method='ywm')   

    if(savepath != None):
        plt.savefig(savepath + "/" + name + ".png")

    plt.show()