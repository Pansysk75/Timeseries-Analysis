from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from typing import Literal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from nolitsa import d2, delay, dimension
import nolds
from statsmodels.tsa.arima.model import ARIMA

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
    ax.set_title('Ljung-Box Portmanteau test p-values')
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


def correlationdimension(xV, m_max, show=False, name = None):
    m_all = np.arange(1, m_max + 1)
    corrdimV = []
    logrM = []
    logCrM = []
    polyM = []

    for m in m_all:
        corrdim, *corrData = nolds.corr_dim(xV, m, debug_data=True)
        corrdimV.append(corrdim)
        logrM.append(corrData[0][0])
        logCrM.append(corrData[0][1])
        polyM.append(corrData[0][2])
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(m_all, corrdimV, marker='x', linestyle='-.')
        ax.set_xlabel('m')
        ax.set_xticks(m_all)
        ax.set_ylabel('v')
        ax.set_title(f'Corr Dim vs m {name}')

        plt.savefig(f"./Part2/plots/cor_dim_{name}.png")
        plt.show()
        plt.close()

    return corrdimV, logrM, logCrM, polyM

def plot_correlation_dimension(xV, tau, m_max, name = None):
    plt.figure(figsize=(6, 4))
    plt.title(f'Local $D_2$ vs $r$ for {name}')
    plt.xlabel(r'Distance $r$')
    plt.ylabel(r'Local $D_2$')
    dim = np.arange(1, m_max + 1)
    m = dim[0]

    for r, c in d2.c2_embed(xV, tau= tau, dim=dim):
        plt.semilogx(r[3:-3], d2.d2(r, c), label=f'm={m}')
        m += 1

    plt.legend()
    plt.savefig(f"./Part2/plots/cordim_{name}.png")
    plt.show()


def plot_correlation_dimension_2(xV, m_max = 10, rmin=0.1, rmax=100, name = None):
    # Plot again using different method
    debug_dataM = []
    corr_dimM = []
    dim = np.arange(1, m_max + 1)
    rvals = np.logspace(np.log(rmin), np.log(rmax), 50)
    for m in dim:
        corr_dim, debug_data = nolds.corr_dim(xV, emb_dim=m, rvals=rvals, debug_data=True)
        debug_dataM.append(debug_data)
        corr_dimM.append(corr_dim)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i, debug_data in enumerate(debug_dataM):
        rvals = debug_data[0] #values used for log(r)
        csums = debug_data[1] #the corresponding log(C(r))
        poly = debug_data[2] #line coefficients ([slope, intercept])
        ax[0].plot(rvals, csums, label=f'm={i+1}')
        ax[2].plot(rvals[1:], np.diff(csums)/np.diff(rvals), label=f'm={i+1}')
        # poly_y = [poly[0] * xi + poly[1] for xi in rvals]
        # ax[0].plot(rvals, poly_y, label=f'POLYY m={i+1}')

    ax[0].set_xlabel('log(r)')
    ax[0].set_xscale("log")
    ax[0].set_ylabel('log(C(r))')
    ax[1].plot(dim, corr_dimM, label='v')
    ax[1].set_xlabel('m')
    ax[1].set_ylabel('v')
    ax[2].set_xlabel('log(r)')
    ax[2].set_xscale("log")
    ax[2].set_ylabel('slope')
    plt.title(f'Local $D_2$ vs $r$ for {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Part2/plots/cordim2_{name}.png")
    plt.show()


def nrmse(trueV, predictedV):
    vartrue = np.sum((trueV - np.mean(trueV)) ** 2)
    varpred = np.sum((predictedV - trueV) ** 2)
    return np.sqrt(varpred / vartrue)




def localpredictnrmse(xV, nlast, m, tau=1, Tmax=1, nnei=1, q=0, show=True, timeseries_name=None):
    xV = xV.reshape(-1, )
    n = xV.shape[0]
    if nlast > n - 2 * m * tau:
        print('test set too large')
    n1 = n - nlast
    if n1 < 2 * (m - 1) * tau - Tmax:
        print('the length of training set is too small for this data size')
    n1vec = n1 - (m - 1) * tau - 1
    xM = np.full(shape=(n1vec, m), fill_value=np.nan)
    for j in np.arange(m):
        xM[:, m - j - 1] = xV[j * tau:n1vec + j * tau]
    from scipy.spatial import KDTree
    kdtreeS = KDTree(xM)

    # For each target point, find neighbors, apply the linear models and keep track
    # of the predicted values each prediction time.
    ntar = nlast - Tmax + 1
    preM = np.full(shape=(ntar, Tmax), fill_value=np.nan)
    winnowM = np.full(shape=(ntar, (m - 1) * tau + 1), fill_value=np.nan)

    ifirst = n1 - (m - 1) * tau
    for i in np.arange((m - 1) * tau + 1):
        winnowM[:, i] = xV[ifirst + i - 1:ifirst + ntar + i - 1]

    for T in np.arange(1, Tmax + 1):
        targM = winnowM[:, :-(m + 1) * tau:-tau]
        _, nneiindM = kdtreeS.query(targM, k=nnei, p=2)
        for i in np.arange(ntar):
            neiM = xM[nneiindM[i], :]
            yV = xV[nneiindM[i] + (m - 1) * tau + 1]
            if q == 0 or nnei == 1:
                preM[i, T - 1] = np.mean(yV)
            else:
                mneiV = np.mean(neiM, axis=0)
                my = np.mean(yV)
                zM = neiM - mneiV
                [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)
                Sx = np.diag(Sx)
                Vx = Vx.T
                tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)
                lsbV = tmpM @ (yV - my)
                preM[i, T - 1] = my + (targM[i, :] - mneiV) @ lsbV
        winnowM = np.concatenate([winnowM, preM[:, [T - 1]]], axis=1)
    nrmseV = np.full(shape=(Tmax, 1), fill_value=np.nan)

    start_idx = (n1vec + (m - 1) * tau)
    end_idx = start_idx + preM.shape[0]
    for t_idx in np.arange(1, Tmax + 1):
        nrmseV[t_idx - 1] = nrmse(trueV=xV[start_idx + t_idx:end_idx + t_idx], predictedV=preM[:, t_idx - 1])
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, Tmax + 1), nrmseV, marker='x')
        ax.set_xlabel('prediction time T')
        ax.set_ylabel('NRMSE(T)')
        ax.axhline(1., color='yellow')
        ax.set_title(f'NRMSE(T), {timeseries_name} m={m}, tau={tau}, q={q}, n={n}, nlast={nlast}')

        plt.savefig(f"./Part2/plots/nrmse_local_q{q}_{timeseries_name}.png")
        plt.show()
        plt.close()

    return nrmseV, preM
