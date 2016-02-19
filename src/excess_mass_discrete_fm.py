import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy.optimize import brentq, minimize
#from scipy.integrate import quad, quadrature
from scipy.stats import norm
#from scipy.stats import gaussian_kde
from ApproxGaussianKDE import ApproxGaussianKDE as KDE

import matplotlib.pyplot as plt


def testfun(x):
    return x

try:
    testfun = profile(testfun)
except NameError:
    def profile(fun):
        return fun


@profile
def fisher_marron_statistic(data, k, lambda0, m0, htol=1e-3):
    h_k = critical_bandwith(data, k, lambda0, m0, htol)
    F_hk_X = probability_integral_transform(data, h_k)
    return von_mises(F_hk_X)


@profile
def fisher_marron_statistic_hk(data, h_k):
    F_hk_X = probability_integral_transform(data, h_k)
    return von_mises(F_hk_X)


def fisher_marron_pval(data, k, lambda0, m0, B=100, htol=1e-3):
    h_k = critical_bandwith(data, k, lambda0, m0, htol)
    T_k_data = fisher_marron_statistic_hk(data, h_k)
    #print "T_k_data = {}".format(T_k_data)
    sampling_density = KernelDensity(kernel='gaussian', bandwidth=h_k).fit(data)  # gaussian_kde(data, h_k).reshape(-1)  #
    T_bootstrap = np.zeros(B)
    for i in range(B):
        print "\ri = {}".format(i),
        data_bootstrap = sampling_density.sample(len(data))
        T_bootstrap[i] = fisher_marron_statistic(data_bootstrap, k, lambda0, m0, htol)
    #print "\n"
    #print "\nT_bootstrap = {}".format(T_bootstrap)
    return np.mean(T_k_data > T_bootstrap)


def von_mises(U):
    '''
        Assumes U is sorted
    '''
    n = len(U)
    i = np.arange(1, n+1)
    return np.sum((U - (2*i-1)/(2*n))**2) + 1./(12*n)


@profile
def probability_integral_transform(X, h_k):
    #kde = KernelDensity(kernel='gaussian', bandwidth=h_k).fit(X)  # gaussian_kde(X.reshape(-1), h_k)
    X = np.sort(X)
    T1 = np.array([kde_distribution_function(x, X, h_k) for x in X])
    # return np.array([kde.integrate_box_1d(X[0] - 3*h_k, x) for x in X])
    # X = np.hstack([X[0]-3*h_k, X.reshape(-1)])
    # T2 = np.cumsum([quadrature(lambda x: np.exp(kde.score_samples(x.reshape(-1, 1))),
    #                             X_left, X_right, rtol=1e-5, tol=1e-5)[0]
    #                  for (X_left, X_right) in zip(X[:-1], X[1:])])
    # print "kde_distribution_function(X[0], X[1:], h_k) = {}".format(kde_distribution_function(X[0], X[1:], h_k))
    # print "T1-T2 = {}".format(T1-T2)
    return T1


def kde_distribution_function(x, data, h_k):
    return np.mean(norm.cdf(x - data, scale=h_k))


def critical_bandwith(data, k, lambda0, m0, htol=1e-3):
    hmax = np.max(data)-np.min(data)
    if positive_Sk(hmax, data, k, lambda0, m0) > 0:
        raise ValueError('Data has more than k modes for maximal bandwidth')
    return binary_search(positive_Sk, 0, hmax, htol, data, k, lambda0, m0)
    # hmin = hmax*1./2
    # while positive_Sk(hmin, data, k, lambda0, m0) < 0:
    #     hmax = hmin
    #     hmin /= 2
    # return brentq(positive_Sk, hmin, hmax,
    #               args=(data, k, lambda0, m0), xtol=htol)


def binary_search(fun, xmin, xmax, tol, *args):
    '''
        Assuming fun(xmax) < 0.
    '''
    if xmax-xmin < tol:
        return (xmin + xmax)*1./2
    xnew = (xmin + xmax)*1./2
    fxnew = fun(xnew, *args)
    if fxnew < 0:
        return binary_search(fun, xmin, xnew, tol, *args)
    return binary_search(fun, xnew, xmax, tol, *args)


@profile
def positive_Sk(h, data, k, lambda0, m0):
    '''
        Returns 1 if Sk is positive (i.e. we get more than k modes),
        otherwise returns -1.
    '''
    #print "Testing h = {}".format(h)
    mi, ma = min(data), max(data)
    if len(data) > 1000:
        kde = KDE(data.reshape(-1), h)
    else:
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(data)
    dx = h*1./10
    x_grid = np.arange(mi, ma, step=dx).reshape(-1, 1)
    log_prob_x = kde.score_samples(x_grid)
    ind_loc_max = argrelextrema(log_prob_x, np.greater_equal)[0]  # FIXME! Remove adjacent max
    x_loc_max = x_grid[ind_loc_max]
    if len(x_loc_max) <= k:
        return -1

    ## Remove low maxima
    above_lambda0 = log_prob_x[ind_loc_max] > np.log(lambda0)
    x_loc_max = x_loc_max[above_lambda0]
    ind_loc_max = ind_loc_max[above_lambda0]
    if len(x_loc_max) <= k:
        return -1

    x_loc_min = np.zeros(len(x_loc_max)+1)
    x_loc_min[0] = mi - 3*h  # Assuming decreasing density at endpoints
    x_loc_min[-1] = ma + 3*h
    for i, (ind_left, ind_right) in enumerate(zip(ind_loc_max[:-1], ind_loc_max[1:])):
        #x_left = x_grid[ind_left] + dx  # Need to exclude maxima, since might otherwise get stalled there.
        #x_right = x_grid[ind_right] - dx
        x0 = x_grid[ind_left+1:ind_right][np.argmin(log_prob_x[ind_left+1:ind_right])]
        x_loc_min[i+1] = x0  # minimize(lambda x: np.exp(kde.score_samples(x)), x0=x0,
                             #          bounds=[(x_left, x_right)]).x

    # plt.figure()
    # plt.plot(x_grid, np.exp(log_prob_x))
    # [plt.axvline(x=xlm, color='green') for xlm in x_loc_max]
    # [plt.axvline(x=xlm, color='red', ls='--') for xlm in x_loc_min]
    # plt.show()

    ## Locate modes
    K = len(x_loc_max)
    lambdas = np.zeros(K)
    a = np.zeros(K)*np.nan
    b = np.zeros(K)*np.nan
    prob_x_loc_min = np.exp(kde.score_samples(x_loc_min.reshape(-1, 1)))
    ind_loc_min = np.hstack([[0], np.searchsorted(x_grid.ravel(), x_loc_min[1:-1])-1, [len(x_grid)]])

    for i, (pr_left, pr_right) in enumerate(zip(prob_x_loc_min[:-1], prob_x_loc_min[1:])):
        if pr_left > pr_right:
            if pr_left > lambda0:
                lambdas[i] = pr_left
                a[i] = x_loc_min[i]
            else:
                lambdas[i] = lambda0
        else:
            if pr_right > lambda0:
                lambdas[i] = pr_right
                b[i] = x_loc_min[i+1]
            else:
                lambdas[i] = lambda0

        if np.isnan(a[i]):
            ind_right = ind_loc_min[i] + np.searchsorted(log_prob_x[ind_loc_min[i]:ind_loc_max[i]+1], np.log(lambdas[i]))
            if ind_right > 0:
                a[i] = (x_grid[ind_right-1] + x_grid[ind_right])/2
            else:
                a[i] = brentq(lambda x: kde.score_samples(x) - np.log(lambdas[i]),
                              x_loc_min[0], x_grid[0], rtol=1e-6)

        if np.isnan(b[i]):
            ind_right = ind_loc_max[i] + np.searchsorted(-log_prob_x[ind_loc_max[i]:ind_loc_min[i+1]+1], -np.log(lambdas[i]))
            if ind_right < len(x_grid):
                b[i] = (x_grid[ind_right-1] + x_grid[ind_right])/2
            else:
                b[i] = brentq(lambda x: kde.score_samples(x) - np.log(lambdas[i]),
                              x_grid[-1], x_loc_min[-1], rtol=1e-6)
    # plt.figure()
    # plt.plot(x_grid, np.exp(log_prob_x))
    # for a_i in a:
    #     plt.axvline(x=a_i, color='green')
    # for b_i in b:
    #     plt.axvline(x=b_i, color='red')
    # plt.show()

    ## Remove small maxima
    E = np.zeros(K)
    for i, (a_i, b_i, lamb) in enumerate(zip(a, b, lambdas)):
        E[i] = (kde_distribution_function(b_i, data, h) -
                kde_distribution_function(a_i, data, h) - lamb*(b_i-a_i))
        #E[i] = quad(lambda x: np.exp(kde.score_samples(x)) - lamb, a_i, b_i)[0]
    ind_bool = np.ones(K, dtype='bool')
    ind = np.nonzero(ind_bool)[0]
    log_lambdas = np.log(lambdas)
    while np.sum(E[ind] > m0) <= k:
        argmin = np.argmin(E[ind])
        i_E_min = ind[argmin]
        if np.isclose(log_lambdas[i_E_min], kde.score_samples(a[i_E_min])):
            if argmin > 0:
                i_E_min_left = ind[argmin-1]
                E[i_E_min_left] += E[i_E_min]
                b[i_E_min_left] = b[i_E_min]
        elif np.isclose(log_lambdas[i_E_min], kde.score_samples(b[i_E_min])):
            if argmin < len(ind)-1:
                i_E_min_right = ind[argmin+1]
                E[i_E_min_right] += E[i_E_min]
                a[i_E_min_right] = a[i_E_min]
        ind_bool[i_E_min] = False
        ind = np.nonzero(ind_bool)[0]
        if len(ind) <= k:
            return -1

    return 1

if __name__ == '__main__':
    N = 1000
    seed = np.random.randint(0, 1000)
    seed = 286
    print "seed = {}".format(seed)
    np.random.seed(seed)
    dat = np.hstack([np.random.randn(N/2), np.random.randn(N/2)+4]).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(dat)
    xx = np.arange(-4, 8, step=.1).reshape(-1, 1)
    plt.plot(xx, np.exp(kde.score_samples(xx)))

    # h_1 = critical_bandwith(dat, 1, 0.1, 0.00001)
    # kde = KernelDensity(kernel='gaussian', bandwidth=h_1).fit(dat)
    # xx = np.arange(-4, 8, step=.1).reshape(-1, 1)
    # plt.plot(xx, np.exp(kde.score_samples(xx)))

    #h_2 = critical_bandwith(dat, 2, 0.1, 0.0000001)
    # kde = KernelDensity(kernel='gaussian', bandwidth=h_2).fit(dat)
    # xx = np.arange(-4, 8, step=.1).reshape(-1, 1)
    # plt.plot(xx, np.exp(kde.score_samples(xx)))

    # h_3 = critical_bandwith(dat, 3, 0.01, 0.0000001)
    # kde = KernelDensity(kernel='gaussian', bandwidth=h_3).fit(dat)
    # xx = np.arange(-4, 8, step=.1).reshape(-1, 1)
    # plt.plot(xx, np.exp(kde.score_samples(xx)))

    #plt.show()

    import time
    t0 = time.time()
    print "fisher_marron_statistic(dat, 1, 0.1, 0.00001) = {}".format(fisher_marron_statistic(dat, 1, 0.1, 0.00001))
    t1 = time.time()
    print "Statistic time: {}".format(t1-t0)
    print "fisher_marron_pval(dat, 1, 0.1, 0.01, B=10) = {}".format(fisher_marron_pval(dat, 1, 0.1, 0.01, B=10))
    t2 = time.time()
    print "Pval time: {}".format(t2-t1)