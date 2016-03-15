import numpy as np
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

from .util.ApproxGaussianKDE import ApproxGaussianKDE as KDE
from .calibration.lambda_alphas_access import load_lambda


def pval_silverman(data, I=(-np.inf, np.inf), N_bootstrap=1000):
    h_crit = critical_bandwidth(data, I)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    smaller_equal_crit_bandwidth = np.zeros((N_bootstrap,), dtype=np.bool_)
    for n in range(N_bootstrap):  # CHECK: Rescale bootstrap sample towards the mean? Yes.
        smaller_equal_crit_bandwidth[n] = is_unimodal_kde(
            h_crit, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data))
    return np.mean(~smaller_equal_crit_bandwidth)


def reject_null_calibrated_test_bandwidth(data, alpha, null, I=(-np.inf, np.inf), N_bootstrap=1000):
    lambda_alpha = load_lambda('bw', null, alpha)
    h_crit = critical_bandwidth(data)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    smaller_equal_crit_bandwidth = np.zeros((N_bootstrap,), dtype=np.bool_)
    for n in range(N_bootstrap):  # CHECK: Rescale bootstrap sample towards the mean? Yes.
        smaller_equal_crit_bandwidth[n] = is_unimodal_kde(
            h_crit*lambda_alpha, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data))
    return np.mean(~smaller_equal_crit_bandwidth) <= alpha


def critical_bandwidth(data, I=(-np.inf, np.inf), htol=1e-3):
        # I is interval over which density is tested for unimodality
    hmax = (np.max(data)-np.min(data))/2.0
    return bisection_search_unimodal(0, hmax, htol, data, I)


def critical_bandwidth_m_modes(data, m, I=(-np.inf, np.inf), htol=1e-3):
        # I is interval over which density is tested for unimodality
    hmax = (np.max(data)-np.min(data))/2.0
    return bisection_search_most_m_modes(0, hmax, htol, data, m, I)


def bisection_search_unimodal(hmin, hmax, htol, data, I):
    '''
        Assuming fun(xmax) < 0.
    '''
    return bisection_search_most_m_modes(hmin, hmax, htol, data, 1, I)


def bisection_search_most_m_modes(hmin, hmax, htol, data, m, I):
    '''
        Assuming fun(xmax) < 0.
    '''
    if hmax-hmin < htol:
        return (hmin + hmax)/2.0
    hnew = (hmin + hmax)/2.0
    #print "hnew = {}".format(hnew)
    if kde_has_at_most_m_modes(hnew, data, m, I):  # upper bound for bandwidth
        return bisection_search_most_m_modes(hmin, hnew, htol, data, m, I)
    return bisection_search_most_m_modes(hnew, hmax, htol, data, m, I)


def is_unimodal_kde(h, data, I=(-np.inf, np.inf)):
    return kde_has_at_most_m_modes(h, data, 1, I)


def kde_has_at_most_m_modes(h, data, m, I=(-np.inf, np.inf)):
    # I is interval over which density is tested for unimodality
    xtol = h*0.1  # TODO: Compute error given xtol.
    kde = KDE(data, h)
    x_new = np.linspace(max(I[0], np.min(data)), min(I[1], np.max(data)), 10)
    x = np.zeros(0,)
    y = np.zeros(0,)
    while True:
        y_new = kde.evaluate_prop(x_new)
        x = merge_into(x_new, x)
        y = merge_into(y_new, y)
        # fig, ax = plt.subplots()
        # ax.plot(x, y)
        # ax.scatter(x, y, marker='+')
        # ax.scatter(x_new, y_new, marker='+', color='red')
        if len(argrelextrema(np.hstack([[0], y, [0]]), np.greater)[0]) > m:
            return False
        if x[1] - x[0] < xtol:
            return True
        x_new = (x[:-1]+x[1:])/2.0


def merge_into(z_new, z):
    if len(z) == 0:
        return z_new
    z_merged = np.zeros((2*len(z)-1,))
    z_merged[np.arange(0, len(z_merged), 2)] = z
    z_merged[np.arange(1, len(z_merged), 2)] = z_new
    return z_merged

if __name__ == '__main__':

    if 0:
        import time
        N = 1000
        data = np.hstack([np.random.randn(N/2), np.random.randn(N/4)+4])
        h = 0.1
        print "is_unimodal_kde(h, data) = {}".format(is_unimodal_kde(h, data))
        #plt.show()
        t0 = time.time()
        h_crit = critical_bandwidth(data)
        print "critical_bandwidth(data) = {}".format(h_crit)
        t1 = time.time()
        print "pval_silverman(data) = {}".format(pval_silverman(data))
        t2 = time.time()
        print "Critical bandwidth computation time: {}".format(t1-t0)
        print "Silverman test computation time: {}".format(t2-t1)

        fig, ax = plt.subplots()
        ax.hist(data, bins=50, normed=True)
        x_grid = np.linspace(np.min(data)-2, np.max(data)+2, 100)
        ax.plot(x_grid, KDE(data, h_crit).evaluate(x_grid), linewidth=2, color='black')
        plt.show()

    if 0:
        data = np.random.randn(1000)
        h = .5
        print "np.std(data) = {}".format(np.std(data))
        resamp = KernelDensity(kernel='gaussian', bandwidth=h).fit(data).sample(1000)/np.sqrt(1+h**2/np.var(data))
        print "np.std(resamp) = {}".format(np.std(resamp))

    if 0:
        N = 1000
        data = np.hstack([np.random.randn(N/2), np.random.randn(N/4)+4])
        h = 0.1
        print "is_unimodal_kde(h, data) = {}".format(is_unimodal_kde(h, data))
        #plt.show()
        h_crit = critical_bandwidth_m_modes(data, 2)
        x = np.linspace(-3, 8, 200)
        y = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1)).score_samples(x.reshape(-1, 1))
        plt.plot(x, np.exp(y))
        plt.show()

    if 1:
        N = 1000
        data = np.hstack([np.random.randn(N/2), np.random.randn(N/4)+4])
        if 0:
            I = (-1.5, 5.5)
            print "pval_silverman(data, I) = {}".format(pval_silverman(data, I))
            print "reject_null_calibrated_test_bandwidth(data, 0.05, 'normal', I) = {}".format(reject_null_calibrated_test_bandwidth(data, 0.05, 'normal', I))
            print "reject_null_calibrated_test_bandwidth(data, 0.05, 'shoulder', I) = {}".format(reject_null_calibrated_test_bandwidth(data, 0.05, 'shoulder', I))

        x = np.linspace(-3, 8, 200)
        print "x.shape = {}".format(x.shape)
        y = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data.reshape(-1, 1)).score_samples(x.reshape(-1, 1))
        plt.plot(x, np.exp(y))
        plt.show()