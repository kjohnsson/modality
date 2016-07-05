import numpy as np


def knn_density(x, data, k):
    data_sort = np.sort(data)
    I_k = [(data_sort[i], data_sort[i+k-1]) for i in range(len(data_sort)-k+1)]
    #d_k = [I_k_[1] - I_k_[0] for I_k_ in I_k]
    x_d_ks = [[max(x_-I_k_[0], I_k_[1]-x_) for I_k_ in I_k if x_ >= I_k_[0] and x_ <= I_k_[1]] for x_ in x]
    x_d_k = np.array([min(d_ks) if len(d_ks) > 0 else np.nan for d_ks in x_d_ks])
    x_d_k[x < data_sort[0]] = data_sort[k-1] - x[x < data_sort[0]]
    x_d_k[x > data_sort[-1]] = x[x > data_sort[-1]] - data_sort[-k]
    return k/(2*len(data)*x_d_k)


def auto_interval(data, k=5, beta=0.2, xmin=0., xmax=1023., dx=1.):
    dens_bound = beta/(xmax-xmin)
    x = np.arange(xmin, xmax+dx, dx)
    above_bound = x[knn_density(x, data, k) > dens_bound]
    return above_bound[0], above_bound[-1]

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    data = np.random.randn(100)
    x = np.linspace(-3, 3, 200)
    k = 10
    plt.plot(x, knn_density(x, data, k))

    import statsmodels as sm
    faithful = sm.datasets.get_rdataset("faithful")
    data = faithful.data.eruptions
    x = np.linspace(0, 6, 200)
    k = 20
    plt.figure()
    plt.plot(x, knn_density(x, data, k))
    plt.hist(faithful.data.eruptions, np.arange(1.5, 5.5, 0.5), normed=True)
    xmin, xmax = auto_interval(data, xmin=1, xmax=6)
    plt.axvline(xmin)
    plt.axvline(xmax)
