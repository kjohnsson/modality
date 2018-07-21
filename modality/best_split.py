from __future__ import unicode_literals
import numpy as np
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

from .critical_bandwidth import critical_bandwidth_m_modes


def best_split(data, I=(-np.inf, np.inf)):
    '''With bimodal data, finding split at lowest density.'''
    h_crit = critical_bandwidth_m_modes(data, 2, I)
    kde = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    x = np.linspace(max(np.min(data), I[0]), min(np.max(data), I[1]), 200)
    y = np.exp(kde.score_samples(x.reshape(-1, 1)))
    modes = argrelextrema(np.hstack([[0], y, [0]]), np.greater)[0]
    if len(modes) != 2:
        raise ValueError("{} modes at: {}".format(len(modes), x[modes-1]))
    ind_min = modes[0]-1 + argrelextrema(y[(modes[0]-1):(modes[1]-1)], np.less)[0]
    return x[ind_min]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    if 1:
        N = 1000
        data = np.hstack([np.random.randn(N/2), np.random.randn(N/4)+4])
        h_crit = critical_bandwidth_m_modes(data, 2)
        x = np.linspace(-3, 8)
        y = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1)).score_samples(x.reshape(-1, 1))
        fig, ax = plt.subplots()
        ax.plot(x, np.exp(y))
        ax.axvline(best_split(data, (1, 4)), color='red')
        plt.show()
