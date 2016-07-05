import numpy as np
import matplotlib.pyplot as plt

from .diptest import dip_and_closest_unimodal_from_cdf, cum_distr, least_concave_majorant
from .util.LinkedIntervals import LinkedIntervals


# def excess_mass_modes_test(data, w, n):
#     if w is None:
#         w = np.ones(len(data))*1./len(data)
#     x_F, y_F = cum_distr(data, w)
#     dip, (x_uni, y_uni) = dip_and_closest_unimodal_from_cdf(x_F, y_F)
#     print "dip = {}".format(dip)
#     D_max = -np.inf
#     for lambd in np.arange(0.001, 10, 1e-4):
#     #lambd = 0.15
#         #print "lambd = {}".format(lambd)
#         D, modes = excess_mass_modes_lambda(data, w, n, lambd)
#         if D > D_max:
#             D_max = D
#             best_modes = modes
#     print "D_max = {}".format(D_max)
#     return best_modes


def excess_mass_modes(data, w, n, plotting=False):
    if w is None:
        w = np.ones(len(data))*1./len(data)
    x_F, y_F = cum_distr(data, w)
    _, x_lcm, y_lcm = least_concave_majorant(x_F, y_F)
    #print "i_lcm = {}".format(i_lcm)
    #x_lcm, y_lcm = x_F[i_lcm], y_F[i_lcm]
    dip, (x_uni, y_uni) = dip_and_closest_unimodal_from_cdf(x_F, y_F, plotting=plotting)
    print "dip = {}".format(dip)
    #i = np.argmax(diff(y_uni)/diff(x_uni))
    #print "x_uni[i], x_uni[i+1] = {}, {}".format(x_uni[i], x_uni[i+1])
    #print "excess_mass_modes_lambda(data, w, n, diff(y_uni)[i]/diff(x_uni)[i]) = {}".format(excess_mass_modes_lambda(data, w, n, diff(y_uni)[i]/diff(x_uni)[i]))
    D_max = -np.inf
    fig, axs = plt.subplots(int(np.ceil((len(y_lcm)-1)*1./5)), 5)
    for lambd, ax in zip(diff(y_lcm)/diff(x_lcm), axs.ravel()):
        if np.isinf(lambd):
            continue
        print "lambd = {}".format(lambd)
    #lambd = 0.15
        #print "lambd = {}".format(lambd)
        D, modes = excess_mass_modes_lambda(data, w, n, lambd, ax)
        if D > D_max:
            D_max = D
            best_modes = modes
    print "D_max = {}".format(D_max)
    return best_modes


def excess_mass_modes_lambda(data, w, n, lambd, ax=None):
    order = np.argsort(data)
    H_lambda = np.cumsum(w[order]) - lambd*data[order]
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(data[order], H_lambda)
    intervals = LinkedIntervals((0, len(order)))
    while len(intervals) < 2*n+1:
        #print "[interval.data for interval in intervals] = {}".format([interval.data for interval in intervals])
        E_max_next = -np.inf
        for i, interval_item in enumerate(intervals):
            if np.mod(i, 2) == 0:
                E_new, interval_new = find_mode(H_lambda, *interval_item.data)
            else:
                E_new, interval_new = find_antimode(H_lambda, *interval_item.data)
            if E_new > E_max_next:
                E_max_next = E_new
                interval_best = interval_new
                i_best = i
        intervals.split(i_best, interval_best)
    #print "[interval.data for interval in intervals] = {}".format([interval.data for interval in intervals])
    return E_max_next, [(order[interval.data[0]], order[interval.data[1]-1]) for i, interval in enumerate(intervals) if np.mod(i, 2) == 1]


def find_mode(H_lambda, i_lower, i_upper):
    if i_lower == i_upper:
        return 0, (i_lower, i_upper)
    H_min = np.inf
    E_max = -np.inf
    for i in range(i_lower, i_upper):
        if H_lambda[i] < H_min:
            H_min = H_lambda[i]
            i_mode_lower_curr = i
        E_new = H_lambda[i]-H_min
        if E_new > E_max:
            E_max = E_new
            i_mode_upper = i+1
            i_mode_lower = i_mode_lower_curr
    return E_max, (i_mode_lower, i_mode_upper)


def find_antimode(H_lambda, i_lower, i_upper):
    E, interval = find_mode(-H_lambda, i_lower, i_upper)
    return -E, interval


def diff(x):
    N = len(x)
    dmat = np.eye(N) + np.diag(np.ones(N-1), -1)
    return dmat[1:, :].dot(x)
