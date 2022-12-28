from typing import Literal
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pmd
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


def split_dataset(xV, split_ratio = 0.8):
    split_idx = round(len(xV) * split_ratio)
    return (xV[0:split_idx], xV[split_idx+1:])

def rmse(xV1, xV2):
    return np.sqrt(np.square(np.subtract(xV1, xV2)).mean())

def nrmse(xV1, xV2):
    '''Normalized rmse (normalized by std of xV1)'''
    return rmse(xV1,xV2)/np.std(xV1)

def plot_timeseries_stats(xV, name, savepath=None):
    '''Plot timeseries, auto-correlation and partial auto-correlation.'''
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(name, fontsize=16)

    plt.subplot(2, 1, 1).plot(xV)
    # plt.savefig("plots/timeseries/" + dataset_name + ".png")

    # From statsmodels module.
    plot_acf(xV, zero=False, lags=10, ax=plt.subplot(2,2,3))

    # From statsmodel module (yule-walker method).
    plot_pacf(xV, zero=False, lags=10, ax=plt.subplot(2,2,4), method='ywm')   

    if(savepath != None):
        plt.savefig(savepath + "/" + name + ".png")

    plt.show()

def batch_arima_test(xV, p_min=0, p_max=5, q_min=0, q_max=5, d=0, show=False):
    '''Fits many arima models and returns the one with
    the smalles aic. Returns a statsmodels.tsa.ARIMA model and 
    a list of dictionaries containing "p", "q" and "aic" '''
    aic_values = []
    best_model = None
    best_aic = float('inf')
    for p in range(p_min, p_max+1):
        for q in range(q_min, q_max+1):
            sm.tsa.arima
            model = sm.tsa.ARIMA(xV, order=(p, d, q)).fit()
            aic = model.aicc
            aic_values.append({"p":p, "q":q, "aic":aic})
            print(f"Fitted ({p}, {d}, {q}), aic={aic}")
            if aic < best_aic:
                best_aic = aic
                best_model = model
    if show:
        plt.figure()
        df = pd.DataFrame(aic_values).pivot(index="p", columns="q", values="aic")
        ax = sns.heatmap(df, fmt=".1f",
          norm=matplotlib.colors.PowerNorm(0.3),
        cmap="gray_r",annot=True)
        ax.invert_yaxis()
        plt.show()
    
    return best_model, aic_values


# def generate_model_report(arima_model, train_data):
#     arima_model.predict