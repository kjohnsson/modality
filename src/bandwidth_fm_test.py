import numpy as np
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

from .util.ApproxGaussianKDE import ApproxGaussianKDE as KDE
from .util.bootstrap_MPI import bootstrap_mpi as bootstrap
from .util import MC_error_check


def testfun(x):
    return x

try:
    testfun = profile(testfun)
except NameError:
    def profile(fun):
        return fun


def bandwidth_test_pval(data, lamtol, mtol, N_bootstrap=1000):
    lambda_alpha = 1  # TODO: Replace with correct value according to Cheng & Hall methodology
    h_crit = fisher_marron_critical_bandwidth(data, lamtol, mtol)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resampling_scale_factor = 1.0/np.sqrt(1+h_crit**2/np.var(data))
    smaller_equal_crit_bandwidth = np.zeros((N_bootstrap,), dtype=np.bool_)
    for n in range(N_bootstrap):  # CHECK: Rescale bootstrap sample towards the mean?
        smaller_equal_crit_bandwidth[n] = is_resampled_unimodal_kde(
            KDE_h_crit, resampling_scale_factor, len(data), h_crit*lambda_alpha, lamtol, mtol)
    return np.mean(~smaller_equal_crit_bandwidth)


@MC_error_check
def bandwidth_test_pval_mpi(data, lamtol, mtol, N_bootstrap=1000):
    lambda_alpha = 1  # TODO: Replace with correct value according to Cheng & Hall methodology
    h_crit = fisher_marron_critical_bandwidth(data, lamtol, mtol)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resampling_scale_factor = 1.0/np.sqrt(1+h_crit**2/np.var(data))
    smaller_equal_crit_bandwidth = bootstrap(
        is_resampled_unimodal_kde, N_bootstrap, KDE_h_crit,
        resampling_scale_factor, len(data), h_crit*lambda_alpha, lamtol, mtol)
    return np.mean(~smaller_equal_crit_bandwidth)


def fisher_marron_critical_bandwidth(data, lamtol, mtol, htol=1e-3):
    hmax = (np.max(data)-np.min(data))/2.0
    return bisection_search_unimodal(0, hmax, htol, data, lamtol, mtol)


def bisection_search_unimodal(hmin, hmax, htol, data, lamtol, mtol):
    '''
        Assuming fun(xmax) < 0.
    '''
    if hmax-hmin < htol:
        return (hmin + hmax)/2.0
    hnew = (hmin + hmax)/2.0
    #print "hnew = {}".format(hnew)
    if is_unimodal_kde(hnew, data, lamtol, mtol):  # upper bound for bandwidth
        return bisection_search_unimodal(hmin, hnew, htol, data, lamtol, mtol)
    return bisection_search_unimodal(hnew, hmax, htol, data, lamtol, mtol)


def is_resampled_unimodal_kde(kde, resampling_scale_factor, n, h, lamtol, mtol):
    return is_unimodal_kde(h, kde.sample(n).ravel()*resampling_scale_factor, lamtol, mtol)


@profile
def is_unimodal_kde(h, data, lamtol, mtol, debug=False):
    xtol = h*0.1  # TODO: Compute error given xtol.
    kde = KDE(data, h)
    lamtol_prop = lamtol*kde._norm_factor  # scaled with same proportionality constant as kde
    #print "lamtol_prop = {}".format(lamtol_prop)
    x_new = np.linspace(np.min(data), np.max(data), 40)
    x = np.zeros(0,)
    y = np.zeros(0,)
    zero = np.zeros(1,)
    while True:
        if len(x) > 0:
            if x[1] - x[0] < xtol:
                return True
        y_new = kde.evaluate_prop(x_new)
        x = merge_into(x_new, x)
        y = merge_into(y_new, y)
        if debug:
            fig, axs = plt.subplots(1, 2)
            for ax in axs:
                ax.plot(x, y/kde._norm_factor)
                ax.scatter(x, y/kde._norm_factor, marker='+')
                ax.scatter(x_new, y_new/kde._norm_factor, marker='+', color='red')
        ind_mode = argrelextrema(np.hstack([zero, y, zero]), np.greater)[0]-1

        x_new = (x[:-1]+x[1:])/2.0  # will be used in next iteration

        if len(ind_mode) > 1:  # check if nodes are above tolerance level
            low_modes = y[ind_mode] < lamtol_prop
            nbr_modes = len(ind_mode)
            if nbr_modes - np.sum(low_modes) <= 1:
                continue
            ind_valley = argrelextrema(y, np.less)[0]
            lambdas = np.hstack([[lamtol_prop], y[ind_valley], [lamtol_prop]])
            low_valleys = lambdas < lamtol_prop  # start and end are included as valleys
            low_valleys[-1] = True  # for convenience we wait with setting leftmost to True
            left_boundaries = np.zeros((nbr_modes, nbr_modes+1), dtype=np.int)  # computing left and right boundary for all possible lambda
            right_boundaries = np.zeros((nbr_modes, nbr_modes+1), dtype=np.int)
            left_boundaries[0, ~low_valleys] = np.searchsorted(y[:ind_mode[0]], lambdas[~low_valleys])
            for i in range(1, nbr_modes):
                left_boundaries[i, ~low_valleys] = np.searchsorted(y[ind_valley[i-1]:ind_mode[i]], lambdas[~low_valleys])+ind_valley[i-1]
            for i in range(nbr_modes-1):
                right_boundaries[i, ~low_valleys] = np.searchsorted(-y[ind_mode[i]:ind_valley[i]], -lambdas[~low_valleys])+ind_mode[i]-1
            right_boundaries[-1, :] = np.searchsorted(-y[ind_mode[-1]:], -lambdas)+ind_mode[-1]-1  # ensure that computed mode size is not larger than real

            left_boundaries[:, low_valleys] = left_boundaries[:, np.newaxis, 0]
            right_boundaries[:, low_valleys] = right_boundaries[:, np.newaxis, 0]

            low_valleys[0] = True

            lambda_ind = range(nbr_modes) + (lambdas[:-1] < lambdas[1:])
            lambda_ind[low_valleys[1:]*low_valleys[:-1]] = 0

            for i in np.arange(nbr_modes)[~low_modes]:
                li = lambda_ind[i]
                if debug:
                    axs[0].plot([x[left_boundaries[i, li]], x[right_boundaries[i, li]]], [lambdas[li]/kde._norm_factor]*2)

            # Computing size of individual modes.
            big_enough = np.zeros((nbr_modes,), dtype='bool')
            for i in np.arange(nbr_modes)[~low_modes]:
                li = lambda_ind[i]
                x_left = x[left_boundaries[i, li]]
                x_right = x[right_boundaries[i, li]]
                mode_size = (kde.distr(x_right) - kde.distr(x_left) -
                             lambdas[li]/kde._norm_factor*(x_right-x_left))
                if debug:
                    print "mode_size {} = {}".format(i, mode_size)
                if mode_size > mtol:
                    big_enough[i] = True
                    if np.sum(big_enough) > 1:
                        return False

            # Merging modes
            supermode_ind_start = np.arange(nbr_modes)
            supermode_ind_end = np.arange(nbr_modes)+1
            #lambda_ind = np.arange(nbr_modes)  # which lambdas is used for each supermode
            for i in np.arange(nbr_modes)[~low_modes]:
                if debug:
                    print "i = {}".format(i)
                    print "big_enough[i] = {}".format(big_enough[i])
                start = supermode_ind_start[i]
                end = supermode_ind_end[i]
                while not big_enough[i]:
                    li = start if lambdas[start] > lambdas[end] else end
                    lambd = lambdas[li]
                    if start == 0 and end == nbr_modes:
                        break
                    if low_valleys[start] and low_valleys[end]:
                        break  # isolated mode -- merger not allowed to left or right
                    merge_to_left = (start > 0) and lambdas[start] > lambdas[end]
                    if debug:
                        print "merge_to_left = {}".format(merge_to_left)
                    if merge_to_left or big_enough[end]:
                        break  # merger will not increase number of modes
                    end += 1
                    while lambdas[end] > lambd:
                        end += 1

                    li = start if lambdas[start] > lambdas[end] else end
                    lambd = lambdas[li]
                    x_left = x[left_boundaries[start, li]]
                    x_right = x[right_boundaries[end-1, li]]
                    supermode_size = (kde.distr(x_right) - kde.distr(x_left) -
                                      lambd/kde._norm_factor*(x_right-x_left))
                    if debug:
                        print "supermode_size = {}".format(supermode_size)
                        print "(start, end) = {}".format((start, end))
                        print "(x_left, x_right) = {}".format((x_left, x_right))
                        print "li = {}".format(li)
                    if supermode_size > mtol:
                        big_enough[i] = True
                        for j in range(i):
                            if big_enough[j] and supermode_ind_end[j] > start:
                                big_enough[j] = False
                        if np.sum(big_enough) > 1:
                            if debug:
                                print "(start, end) = {}".format((start, end))
                                axs[1].plot([x_left, x_right], [lambd/kde._norm_factor]*2)
                            return False
                supermode_ind_start[i] = start
                supermode_ind_end[i] = end
                li = start if lambdas[start] > lambdas[end] else end
                x_left = x[left_boundaries[start, li]]
                x_right = x[right_boundaries[end-1, li]]
                if debug:
                    print "(x_left, x_right) = {}".format((x_left, x_right))
                    print "li = {}".format(li)
                    print "(start, end) = {}".format((start, end))
                    lambd = lambdas[li]
                    axs[1].plot([x_left, x_right], [lambd/kde._norm_factor]*2)                


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
        data = np.hstack([np.random.randn(N/2), np.random.randn(N/2)])
        lamtol, mtol = 0.01, 0.01
        #h = 0.1
        #print "is_unimodal_kde(h, data) = {}".format(is_unimodal_kde(h, data))
        #plt.show()
        t0 = time.time()
        print "fisher_marron_critical_bandwidth(data, lamtol, mtol) = {}".format(fisher_marron_critical_bandwidth(data, lamtol, mtol))
        t1 = time.time()
        print "bandwidth_pval(data, lamtol, mtol) = {}".format(bandwidth_test_pval(data, lamtol, mtol, 1000))
        t2 = time.time()
        print "bandwidth_pval_mpi(data, lamtol, mtol) = {}".format(bandwidth_test_pval_mpi(data, lamtol, mtol, 1000))
        t3 = time.time()
        print "Critical bandwidth computation time: {}".format(t1-t0)
        print "Pval computation time: {}".format(t2-t1)
        print "Pval computation time (mpi) = {}".format(t3-t2)

    if 1:
        import time
        N = 1000
        data = np.hstack([np.random.randn(N/2), np.random.randn(N/2)])
        lamtol, mtol = 0.01, 0.01
        t0 = time.time()
        print "bandwidth_pval_mpi(data, lamtol, mtol) = {}".format(bandwidth_test_pval_mpi(data, lamtol, mtol, 10000))
        t1 = time.time()
        print "Time: {}".format(t1-t0)

    if 0:
        data = np.random.randn(1000)
        h = .5
        print "np.std(data) = {}".format(np.std(data))
        resamp = KernelDensity(kernel='gaussian', bandwidth=h).fit(data).sample(1000)/np.sqrt((1+h**2/np.var(data)))
        print "np.std(resamp) = {}".format(np.std(resamp))

    if 0:
        seed = np.random.randint(0, 1000)
        print "seed = {}".format(seed)  # 37, 210, 76, 492, 229
        #seed = 417 #229
        np.random.seed(seed)
        data = np.hstack([np.random.randn(500), np.random.randn(100)+3])
        #data = np.random.randn(1000)
        print "data.shape = {}".format(data.shape)
        h = .2
        lamtol = 0.01
        mtol = 0.01
        print "is_unimodal_kde(h, data, lamtol, mtol) = {}".format(is_unimodal_kde(h, data, lamtol, mtol, debug=False))
        plt.show()





