from mpi4py import MPI
import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import minimize, leastsq
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import norm

from .util.ApproxGaussianKDE import ApproxGaussianKDE as KDE
from .util.bootstrap_MPI import bootstrap, check_equal_mpi
from .util import MC_error_check
from .util.GaussianMixture1d import GaussianMixture1d as GM


def testfun(x):
    return x

try:
    testfun = profile(testfun)
except NameError:
    def profile(fun):
        return fun


def fisher_marron_critical_bandwidth(data, lamtol, mtol, I=(-np.inf, np.inf), htol=1e-3):
    hmax = (np.max(data)-np.min(data))/2.0
    return bisection_search_unimodal(0, hmax, htol, data, lamtol, mtol, I)


def bisection_search_unimodal(hmin, hmax, htol, data, lamtol, mtol, I=(-np.inf, np.inf)):
    '''
        Assuming fun(xmax) < 0.
    '''
    if hmax-hmin < htol:
        return (hmin + hmax)/2.0
    hnew = (hmin + hmax)/2.0
    #print "hnew = {}".format(hnew)
    if is_unimodal_kde(hnew, data, lamtol, mtol, I):  # upper bound for bandwidth
        return bisection_search_unimodal(hmin, hnew, htol, data, lamtol, mtol, I)
    return bisection_search_unimodal(hnew, hmax, htol, data, lamtol, mtol, I)


def is_resampled_unimodal_kde(kde, resampling_scale_factor, n, h, lamtol, mtol, I=(-np.inf, np.inf)):
    return is_unimodal_kde(h, kde.sample(n).ravel()*resampling_scale_factor, lamtol, mtol, I)


def is_unimodal_kde(h, data, lamtol, mtol, I=None, debug=False):
    return len(mode_sizes_kde(h, data, lamtol, mtol, I, debug=debug)) == 0


def is_unimodal_from_kde(kde, lamtol, mtol, I, xtol, debug=False):
    return len(mode_sizes_from_kde(kde, lamtol, mtol, I, xtol, debug)) == 0


def mode_sizes_kde(h, data, lamtol, mtol, I=None, xtol=None, debug=False):
    if xtol is None:
        xtol = step_size_from_mtol(mtol/2)*h
    #print "xtol = {}".format(xtol)
    kde = KDE(data, h)
    if I is None:
        I = np.min(data), np.max(data)
    return mode_sizes_from_kde(kde, lamtol, mtol, I, xtol, debug)


@profile
def mode_sizes_from_kde(kde, lamtol, mtol, I, xtol, debug=False):
    '''
        kde does not necessarily have to be a kernel density estimator.
        It is required that it has the attributes:
            - evaluate_prop: A function that returns the density
              function evaluated at the input vector (or something
              proportional to this).
            - _norm_factor: A normalization factor such that
              kde.evaluate_prop(x)/kde._norm_factor gives the true
              density.
            - distr: A function evaluating the distribution function at
              the input element.
    '''
    lamtol_prop = lamtol*kde._norm_factor  # scaled with same proportionality constant as kde
    #print "lamtol_prop = {}".format(lamtol_prop)
    x_new = np.linspace(I[0], I[1], 40)
    x = np.zeros(0,)
    y = np.zeros(0,)
    zero = np.zeros(1,)
    while True:
        if len(x) > 0:
            if x[1] - x[0] < xtol:
                return np.zeros((0,))
        y_new = kde.evaluate_prop(x_new)
        x = merge_into(x_new, x)
        y = merge_into(y_new, y)
        if debug:
            fig, axs = plt.subplots(1, 2)
            x_plot = np.linspace(x[0], x[-1], 1000)
            y_plot = kde.evaluate_prop(x_plot)
            for ax in axs:
                ax.plot(x_plot, y_plot/kde._norm_factor)
                #ax.plot(x, y/kde._norm_factor)
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
            mode_sizes = np.zeros((nbr_modes,))
            for i in np.arange(nbr_modes)[~low_modes]:
                li = lambda_ind[i]
                x_left = x[left_boundaries[i, li]]
                x_right = x[right_boundaries[i, li]]
                mode_size = (kde.distr(x_right) - kde.distr(x_left) -
                             lambdas[li]/kde._norm_factor*(x_right-x_left))
                if debug:
                    print("mode_size {} = {}".format(i, mode_size))
                if mode_size > mtol:
                    big_enough[i] = True
                    mode_sizes[i] = mode_size
                    if np.sum(big_enough) > 1:
                        return mode_sizes[mode_sizes > 0]

            # Merging modes
            if debug:
                print("Merging modes")
            supermode_ind_start = np.arange(nbr_modes)
            supermode_ind_end = np.arange(nbr_modes)+1
            #lambda_ind = np.arange(nbr_modes)  # which lambdas is used for each supermode
            for i in np.arange(nbr_modes)[~low_modes]:
                if debug:
                    print("i = {}".format(i))
                    print("big_enough[i] = {}".format(big_enough[i]))
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
                        print("merge_to_left = {}".format(merge_to_left))
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
                        print("supermode_size = {}".format(supermode_size))
                        print("(start, end) = {}".format((start, end)))
                        print("(x_left, x_right) = {}".format((x_left, x_right)))
                        print("li = {}".format(li))
                    if supermode_size > mtol:
                        big_enough[i] = True
                        mode_sizes[i] = supermode_size
                        for j in range(i):
                            if big_enough[j] and supermode_ind_end[j] > start:
                                big_enough[j] = False
                        if np.sum(big_enough) > 1:
                            if debug:
                                print("(start, end) = {}".format((start, end)))
                                axs[1].plot([x_left, x_right], [lambd/kde._norm_factor]*2)
                            return mode_sizes[mode_sizes > 0]

                supermode_ind_start[i] = start
                supermode_ind_end[i] = end
                li = start if lambdas[start] > lambdas[end] else end
                x_left = x[left_boundaries[start, li]]
                x_right = x[right_boundaries[end-1, li]]
                if debug:
                    print("(x_left, x_right) = {}".format((x_left, x_right)))
                    print("li = {}".format(li))
                    print("(start, end) = {}".format((start, end)))
                    lambd = lambdas[li]
                    axs[1].plot([x_left, x_right], [lambd/kde._norm_factor]*2)


def find_reference_distr(mtol, shoulder_ratio, shoulder_variance=1, min=0, max=4):
    if shoulder_variance == 1:
        return bisection_search_reference_distr(min, max, mtol, shoulder_ratio, shoulder_variance)

    m_err_tol = 1e-7
    covs = np.array([1, shoulder_variance])
    weights = np.array(shoulder_ratio, dtype=np.float)
    weights /= np.sum(weights)

    def second_mode_size(a):
        mode_sizes = mode_sizes_from_kde(GM(np.array([0, a]), covs, weights),
                                         lamtol=0, mtol=mtol+m_err_tol, I=[-1, a+1], xtol=a/10000)
        if len(mode_sizes) == 0:
            return 0
        return mode_sizes[1]
    minval = 1
    i = 0
    max_init = 200
    while minval > mtol+m_err_tol:
        x0 = min+(max-min)*np.random.rand(1)
        res_minimizer = minimize(second_mode_size, x0=x0, bounds=[(min, max)], tol=1e-10)
        minval = res_minimizer.fun
        #print "(minval, res_minimizer.x, x0) = {}".format((minval, res_minimizer.x, x0))
        i += 1
        if i > max_init:
            raise ValueError('Unimodal function not found')
    min = res_minimizer.x
    return bisection_search_reference_distr(min, max, mtol, shoulder_ratio, shoulder_variance)


def bisection_search_reference_distr(amin, amax, mtol, shoulder_ratio, shoulder_variance, atol=1e-6):
    anew = (amin + amax)/2.0
    if amax - amin < atol:
        return amin  # ensure that resulting distribution is unimodal

    if shoulder_variance == 1:
        data = np.repeat([anew, 0], shoulder_ratio)
        passed = is_unimodal_kde(1, data, 0, mtol)
    else:
        m_err_tol = 1e-7
        covs = np.array([1, shoulder_variance])
        weights = np.array(shoulder_ratio, dtype=np.float)
        weights /= np.sum(weights)
        passed = is_unimodal_from_kde(GM(np.array([0, anew]), covs, weights),
                                      lamtol=0, mtol=mtol+m_err_tol, I=[-1, anew+1], xtol=anew/10000)
    if passed:
        return bisection_search_reference_distr(anew, amax, mtol, shoulder_ratio, shoulder_variance, atol)
    return bisection_search_reference_distr(amin, anew, mtol, shoulder_ratio, shoulder_variance, atol)


def merge_into(z_new, z):
    if len(z) == 0:
        return z_new
    z_merged = np.zeros((2*len(z)-1,))
    z_merged[np.arange(0, len(z_merged), 2)] = z
    z_merged[np.arange(1, len(z_merged), 2)] = z_new
    return z_merged


def bump_size(width):
    return 1-2*norm.cdf(-width/2)-width*norm.pdf(width/2)


def width_from_bump_size(size):
    return leastsq(lambda width: bump_size(width)-size, x0=0.2)[0]


def step_size_from_mtol(mtol):
    t = width_from_bump_size(mtol)
    t_star = width_from_bump_size(0.9*mtol)
    return (t-t_star)/2


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
        print("fisher_marron_critical_bandwidth(data, lamtol, mtol) = {}".format(fisher_marron_critical_bandwidth(data, lamtol, mtol)))
        t1 = time.time()
        print("bandwidth_pval(data, lamtol, mtol) = {}".format(bandwidth_test_pval(data, lamtol, mtol, 1000)))
        t2 = time.time()
        print("bandwidth_pval_mpi(data, lamtol, mtol) = {}".format(bandwidth_test_pval_mpi(data, lamtol, mtol, 1000)))
        t3 = time.time()
        print("Critical bandwidth computation time: {}".format(t1-t0))
        print("Pval computation time: {}".format(t2-t1))
        print("Pval computation time (mpi) = {}".format(t3-t2))

    if 0:
        import time
        N = 1000
        data = np.hstack([np.random.randn(N/2), np.random.randn(N/2)])
        lamtol, mtol = 0.01, 0.01
        t0 = time.time()
        print("bandwidth_pval_mpi(data, lamtol, mtol) = {}".format(bandwidth_test_pval_mpi(data, lamtol, mtol, 10000)))
        t1 = time.time()
        print("Time: {}".format(t1-t0))

    if 0:
        data = np.random.randn(1000)
        h = .5
        print("np.std(data) = {}".format(np.std(data)))
        resamp = KernelDensity(kernel='gaussian', bandwidth=h).fit(data).sample(1000)/np.sqrt((1+h**2/np.var(data)))
        print("np.std(resamp) = {}".format(np.std(resamp)))

    if 0:
        seed = np.random.randint(0, 1000)
        print("seed = {}".format(seed))  # 37, 210, 76, 492, 229)
        #seed = 417 #229
        np.random.seed(seed)
        data = np.hstack([np.random.randn(500), np.random.randn(100)+3])
        #data = np.random.randn(1000)
        print("data.shape = {}".format(data.shape))
        h = .2
        lamtol = 0.01
        mtol = 0.01
        print("is_unimodal_kde(h, data, lamtol, mtol) = {}".format(is_unimodal_kde(h, data, lamtol, mtol, debug=False)))
        plt.show()

    if 0:
        mtol = 0
        x = np.linspace(-2, 5, 200)
        for shoulder_ratio in [(1, 3), (1, 5), (1, 7), (1, 7)]:
            a = find_reference_distr(mtol, shoulder_ratio)
            w2, w1 = shoulder_ratio
            y = w1*np.exp(-x**2/2) + w2*np.exp(-(x-a)**2/2)
            plt.plot(x, y)
        plt.show()

    if 1:
        from shoulder_distributions import bump_distribution
        mtol = 1e-6
        x = np.linspace(-2, 5, 200)
        for shoulder_ratio in [(2, 1), (4, 1)]:
            w = np.array(shoulder_ratio, dtype=np.float)
            w /= np.sum(shoulder_ratio)
            a = bump_distribution(mtol, w, 1)
            I = (-1.5, a+1)
            w1, w2 = shoulder_ratio
            y = w1*np.exp(-x**2/2) + w2*np.exp(-(x-a)**2/2)
            plt.plot(x, y)
            data = np.repeat([0, a], shoulder_ratio)
            print("data = {}".format(data))
            print("mode_sizes_kde(1, data, 0, mtol/2, I) = {}".format(mode_sizes_kde(1, data, 0, mtol/2, I)))
        plt.show()






