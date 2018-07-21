from __future__ import unicode_literals
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from .diptest import dip_and_closest_unimodal_from_cdf, cum_distr
from .util.LinkedIntervals import LinkedIntervals


def excess_mass_modes(data):
    '''
        Finds two most significant modes according to the excess mass
        test (or equivalently the dip test).

        References:

        K. Johnsson and M. Fontes (2016): What is a `unimodal' cell
        population? Investigating the calibrated dip and bandwidth tests
        for unimodality (manuscript).

        Muller and Sawitski (1991): Excess Mass Estimates and Tests for
        Multimodality. JASA. 86(415).

        Cheng and Hall (1998): On mode testing and empirical
        approximations to distributions. Statistics & Probability
        Letters 39.


        Input:
            data    -   one-dimensional data set

        Output:
            Tuple with two intervals given as (lower, upper).
    '''

    return Delta_N2(data)[1]


def Delta_N2(data, w=None):
    '''
        Computes Delta_{len(data), 2}, i.e. excess mass difference, in
        Muller and Sawitski: Excess Mass Estimates and Tests for
        Multimodality. JASA, Vol. 86, No. 415 (Sep., 1991), pp. 738-746

        Input:
            data (N,)   -   data set
            w (N,)      -   data weights

        Value:
            Tuple with two elements:
                1. D_{len(data), n}(lambda)
                2. Intervals C_1, C_2 that give
                   maximal D_{len(data), n}(lambda)

    '''
    lambda_dash = find_lambda_dash(data, w)
    return D_Nn_lambda(data, w, 2, lambda_dash)


def find_lambda_dash(data, w=None):
    '''
        Finding lambda_dash in the proof of Theorem 3.1 in
        Cheng and Hall: On mode testing and empirical approximations to
        distributions. Statistics & Probability Letters 39 (1998) 245-254.
        This is the lambda which gives maximal value of Delta_N2.

        Input:
            data (N,)   -   data set
            w (N,)      -   data weights
    '''
    xF, yF = cum_distr(data, w)
    dip, unimod = dip_and_closest_unimodal_from_cdf(xF, yF)
    xU, yU = unimod
    yUF = np.interp(xF, xU, yU)

    # Where does the unimodal cdf intersect yF+dip and yF-dip?
    i_upp = np.isclose(yUF, yF+dip)
    i_low = np.isclose(yUF, yF-dip)

    # The interesting points are when an intersection with yF+dip is
    # immediately followed by an interstion with yF-dip
    z = i_upp.astype(np.int) - i_low.astype(np.int)
    nonzero = (z > 0) | (z < 0)
    z = z[nonzero]
    interv = np.diff(z) == -2
    i_interv_left = np.arange(len(i_upp))[nonzero][:-1][interv]
    i_interv_right = np.arange(len(i_upp))[nonzero][1:][interv]
    #print zip(i_interv_left, i_interv_right)
    lam = (yF[i_interv_right]-yF[i_interv_left]-2*dip)/(xF[i_interv_right] - xF[i_interv_left])

    # The highest lambda is lambda_star, the lowest is lambda_dash
    lambda_dash = np.min(lam)
    if np.isnan(lambda_dash) or np.isinf(lambda_dash):
        # (xF, yF-dip/2) and (xF, yF+dip/2) touches at same point.
        # => Any lambda smaller than or equal to lambda_star will do
        lambda_dash = np.diff(yU[:2])/np.diff(xU[:2])
    return lambda_dash


def Delta_Nn_best_lambda_in_unimodal(data, w, n, plotting=False):
    '''
        Uses certain reasonable values for lambda to maximize
        D_Nn_lambda, i.e. to find Delta_{len(data), n}, in
        Muller and Sawitski: Excess Mass Estimates and Tests for
        Multimodality. JASA, Vol. 86, No. 415 (Sep., 1991), pp. 738-746.
        Delta_{len(data), n} should be equal to twice times dip.

        Returns tuple where first item is D_{len(data), n}(lambda)
        and second item is the intervals C_1, C_2 that gives
        maximal D_{len(data), n}(lambda).

        Input:
            data (N,)   -   data set
            w (N,)      -   data weights

    '''
    x_F, y_F = cum_distr(data, w)
    dip, (x_uni, y_uni) = dip_and_closest_unimodal_from_cdf(x_F, y_F, plotting=plotting)
    #print "dip = {}".format(dip)
    D_max = -np.inf
    if plotting:
        fig, axs = plt.subplots(int(np.ceil((len(x_uni)-1)*1./5)), 5)
    else:
        axs = [None]*len(x_uni)-1
    for lambd, ax in zip(np.diff(y_uni)/np.diff(x_uni), axs.ravel()):
        if np.isinf(lambd):
            continue
        #print "lambd = {}".format(lambd)
        D, modes = D_Nn_lambda(data, w, n, lambd, ax)
        if D > D_max:
            D_max = D
            best_modes = modes
    print("D_max - 2*dip= {}".format(D_max-2*dip))  # should be zero
    return best_modes


def Delta_Nn_brute_search(data, w, n):
    '''
        Uses brute force search for lambda to maximize
        D_Nn_lambda, i.e. to find Delta_{len(data), n}, in
        Muller and Sawitski: Excess Mass Estimates and Tests for
        Multimodality. JASA, Vol. 86, No. 415 (Sep., 1991), pp. 738-746.
        Delta_{len(data), n} should be equal to twice times dip.

        Returns tuple where first item is D_{len(data), n}(lambda)
        and second item is the intervals C_1, C_2 that gives
        maximal D_{len(data), n}(lambda).

        Input:
            data (N,)   -   data set
            w (N,)      -   data weights
    '''
    x_F, y_F = cum_distr(data, w)
    dip, (x_uni, y_uni) = dip_and_closest_unimodal_from_cdf(x_F, y_F)
    D_max = -np.inf
    for lambd in np.arange(0.001, 10, 1e-4):
    #lambd = 0.15
        #print "lambd = {}".format(lambd)
        D, modes = D_Nn_lambda(data, w, n, lambd)
        if D > D_max:
            D_max = D
            best_modes = modes
    print("D_max - 2*dip = {}".format(D_max-2*dip))
    return best_modes


def D_Nn_lambda(data, w, n, lambd, ax=None):
    '''
        Computes D_{len(data), n}(lambda) in
        Muller and Sawitski: Excess Mass Estimates and Tests for
        Multimodality. JASA, Vol. 86, No. 415 (Sep., 1991), pp. 738-746

        Returns tuple where first item is D_{len(data), n}(lambda)
        and second item is the intervals C_1, ..., C_n that gives
        maximal D_{len(data), n}(lambda).

    '''
    xF, yF = cum_distr(data, w)
    H_lambda = yF - lambd*xF
    if not ax is None:
        ax.plot(xF, H_lambda)
    intervals = LinkedIntervals((0, len(xF)))
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
    return E_max_next, [(xF[interval.data[0]], xF[interval.data[1]-1]) for i, interval
                        in enumerate(intervals) if np.mod(i, 2) == 1]


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
    return E, interval


# def diff(x):
#     N = len(x)
#     dmat = np.eye(N) + np.diag(np.ones(N-1), -1)
#     return dmat[1:, :].dot(x)
