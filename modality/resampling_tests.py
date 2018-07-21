from __future__ import unicode_literals
from mpi4py import MPI
import numpy as np
from sklearn.neighbors import KernelDensity

from .util.bootstrap_MPI import bootstrap, check_equal_mpi, probability_above, MaxSampExceededException
from .util import get_I
from .calibration.lambda_alphas_access import load_lambda
from . import diptest
from .critical_bandwidth import critical_bandwidth, is_unimodal_kde
from .critical_bandwidth_fm import fisher_marron_critical_bandwidth, is_resampled_unimodal_kde


def calibrated_diptest(data, alpha, null, adaptive_resampling=True, N_adaptive_max=10000,
                       N_non_adaptive=1000, comm=MPI.COMM_WORLD, calibration_file=None):
    '''
        Perform diptest calibrated at level alpha.

        References:

        K. Johnsson and M. Fontes (2016): What is a `unimodal' cell
        population? Investigating the calibrated dip and bandwidth tests
        for unimodality (manuscript).

        M.-Y. Cheng and P. Hall (1999): Mode testing in difficult cases.
        The Annals of Statistics. 27(4).

        Input:
            data                -   data set (one-dimensional)
            alpha               -   level for calibration and test
            null                -   'shoulder' or 'normal'. Reference
                                    distribution for calibration.
            adaptive_resampling -   should adaptive resampling be used?
            N_adaptive_max      -   when number of bootstrap samples
                                    exceeds this value, test is undeter-
                                    mined.
            N_non_adaptive      -   number of bootstrap samples if not
                                    adaptive resampling.
            comm                -   communicator for MPI
            calibration_file    -   file with calibration constants. If
                                    None, precomputed constants are
                                    used.

        Value:
            If adaptive_resampling=True:
                0 when unimodality is rejected.
                1 when unimodality is determined to be not rejected.
                alpha when test is undetermined.

            If adaptive_resampling=False:
                p-value for test for unimodality. NB! The level of the
                test is calibrated correctly for p=alpha.
    '''
    if adaptive_resampling:
        return test_calibrated_dip_adaptive_resampling(
            data, alpha, null, N_adaptive_max, comm, calibration_file)
    return pval_calibrated_dip(
        data, alpha, null, N_non_adaptive, comm, calibration_file)


def calibrated_bwtest(data, alpha, null, I='auto', adaptive_resampling=True,
                      N_adaptive_max=10000, N_non_adaptive=1000, comm=MPI.COMM_WORLD,
                      calibration_file=None):
    '''
        Perform bandwidth test calibrated at level alpha.

        References:

        K. Johnsson and M. Fontes (2016): What is a `unimodal' cell
        population? Investigating the calibrated dip and bandwidth tests
        for unimodality (manuscript).

        M.-Y. Cheng and P. Hall (1999): Mode testing in difficult cases.
        The Annals of Statistics. 27(4).

        Input:
            data                -   data set (one-dimensional)
            alpha               -   level for calibration and test
            null                -   'shoulder' or 'normal'. Reference
                                    distribution for calibration.
            I                   -   interval in which modes are counted
                                    or None (equiv. to I=(-np.inf, np.inf))
                                    or 'auto' (equiv. to I=auto_interval(data))
                                    or {type: 'auto', par_to_auto_interval},
                                    (equiv. to I=auto_interval(data,**par_to_auto_interval))
            adaptive_resampling -   should adaptive resampling be used?
            N_adaptive_max      -   when number of bootstrap samples
                                    exceeds this value, test is undeter-
                                    mined.
            N_non_adaptive      -   number of bootstrap samples if not
                                    adaptive resampling.
            calibration_file    -   file with calibration constants. If
                                    None, precomputed constants are
                                    used.

        Value:
            If adaptive_resampling=True:
                0 when unimodality is rejected.
                1 when unimodality is determined to be not rejected.
                alpha when test is undetermined.

            If adaptive_resampling=False:
                p-value for test for unimodality. NB! The level of the
                test is calibrated correctly for p=alpha.
    '''
    if adaptive_resampling:
        return test_calibrated_bandwidth_adaptive_resampling(
            data, alpha, null, I, N_adaptive_max, comm, calibration_file)
    return pval_calibrated_bandwidth(
        data, alpha, null, I, N_non_adaptive, comm, calibration_file)


def silverman_bwtest(data, alpha, I='auto', adaptive_resampling=True, N_adaptive_max=10000,
                     N_non_adaptive=1000, comm=MPI.COMM_WORLD):
    '''
        Perform Silverman's bandwidth test.

        References:

        K. Johnsson and M. Fontes (2016): What is a `unimodal' cell
        population? Investigating the calibrated dip and bandwidth tests
        for unimodality (manuscript).

        Silverman (1981): Using kernel density estimates to
        investigate multimodality. Journal of the Royal Statistical
        Society. Series B. 43(1).

        Input:
            data                -   data set (one-dimensional)
            alpha               -   significance level
            I                   -   interval in which modes are counted
                                    or None (equiv. to I=(-np.inf, np.inf))
                                    or 'auto' (equiv. to I=auto_interval(data))
                                    or {type: 'auto', par_to_auto_interval},
                                    (equiv. to I=auto_interval(data,**par_to_auto_interval))
            adaptive_resampling -   should adaptive resampling be used?
            N_adaptive_max      -   when number of bootstrap samples
                                    exceeds this value, test is undeter-
                                    mined.
            N_non_adaptive      -   number of bootstrap samples if not
                                    adaptive resampling.

        Value:
            If adaptive_resampling=True:
                0 when unimodality is rejected.
                1 when unimodality is determined to be not rejected.
                alpha when test is undetermined.

            If adaptive_resampling=False:
                p-value for test for unimodality.
    '''

    if adaptive_resampling:
        return test_silverman_adaptive_resampling(data, alpha, I, N_adaptive_max, comm)
    return pval_silverman(data, I, N_non_adaptive, comm)


def test_calibrated_dip_adaptive_resampling(data, alpha, null, N_bootstrap_max=10000,
                                            comm=MPI.COMM_WORLD, calibration_file=None):
    data = comm.bcast(data)
    try:
        lambda_alpha = load_lambda('dip_ad', null, alpha, calibration_file)
           # loading lambda computed with adaptive probablistic bisection search
    except KeyError:
        lambda_alpha = load_lambda('dip_ex', null, alpha, calibration_file)
           # loading lambda computed with probabilistic bisection search
    xF, yF = diptest.cum_distr(data)
    dip, unimod = diptest.dip_and_closest_unimodal_from_cdf(xF, yF)
    resamp_fun = lambda: diptest.dip_resampled_from_unimod(unimod, len(data)) > lambda_alpha*dip
    try:
        return float(probability_above(resamp_fun, alpha, max_samp=N_bootstrap_max, comm=comm,
                     batch=100, bound_significance=0.05, exception_at_max_samp=True,
                     printing=False))
    except MaxSampExceededException:
        return alpha


def test_calibrated_bandwidth_adaptive_resampling(data, alpha, null, I='auto',
                                                  N_bootstrap_max=10000, comm=MPI.COMM_WORLD,
                                                  calibration_file=None):
    data = comm.bcast(data)
    I = get_I(data, I)
    try:
        lambda_alpha = load_lambda('bw_ad', null, alpha, calibration_file)
           # loading lambda computed with adaptive probablistic bisection search
    except KeyError:
        lambda_alpha = load_lambda('bw', null, alpha, calibration_file)
           # loading lambda computed with probabilistic bisection search
    h_crit = critical_bandwidth(data, I)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resamp_fun = lambda: not is_unimodal_kde(
        h_crit*lambda_alpha, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data), I)
    try:
        return float(probability_above(resamp_fun, alpha, max_samp=N_bootstrap_max, comm=comm,
                     batch=100, bound_significance=0.05, exception_at_max_samp=True,
                     printing=False))
    except MaxSampExceededException:
        return alpha


def test_silverman_adaptive_resampling(data, alpha, I='auto',
                                       N_bootstrap_max=10000, comm=MPI.COMM_WORLD):
    data = comm.bcast(data)
    I = get_I(data, I)
    h_crit = critical_bandwidth(data, I)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resamp_fun = lambda: not is_unimodal_kde(
        h_crit, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data), I)
    try:
        return float(probability_above(resamp_fun, alpha, max_samp=N_bootstrap_max, comm=comm,
                     batch=100, bound_significance=0.05, exception_at_max_samp=True,
                     printing=False))
    except MaxSampExceededException:
        return alpha


def pval_calibrated_dip(data, alpha_cal, null, N_bootstrap=1000, comm=MPI.COMM_WORLD,
                        calibration_file=None):
    '''
        NB!: Test is only calibrated to correct level for alpha_cal.
    '''
    data = comm.bcast(data)
    try:
        lambda_alpha = load_lambda('dip_ad', null, alpha_cal, calibration_file)
    except KeyError:
        lambda_alpha = load_lambda('dip_ex', null, alpha_cal, calibration_file)
    xF, yF = diptest.cum_distr(data)
    dip, unimod = diptest.dip_and_closest_unimodal_from_cdf(xF, yF)
    resamp_fun = lambda: diptest.dip_resampled_from_unimod(unimod, len(data))
    resamp_dips = bootstrap(resamp_fun, N_bootstrap, dtype=np.float_, comm=comm)
    return np.mean(resamp_dips > lambda_alpha*dip)


def pval_silverman(data, I='auto', N_bootstrap=1000, comm=MPI.COMM_WORLD):
    I = get_I(data, I)
    data = comm.bcast(data)
    h_crit = critical_bandwidth(data, I)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resamp_fun = lambda: is_unimodal_kde(
        h_crit, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data), I)
    smaller_equal_crit_bandwidth = bootstrap(resamp_fun, N_bootstrap, dtype=np.bool_,
                                             comm=comm)
    return np.mean(~smaller_equal_crit_bandwidth)


def pval_calibrated_bandwidth(data, alpha_cal, null, I='auto',
                              N_bootstrap=1000, comm=MPI.COMM_WORLD,
                              calibration_file=None):
    '''
        NB!: Test is only calibrated to correct level for alpha_cal.
    '''
    data = comm.bcast(data)
    I = get_I(data, I)
    try:
        lambda_alpha = load_lambda('bw_ad', null, alpha_cal, calibration_file)
    except KeyError:
        lambda_alpha = load_lambda('bw', null, alpha_cal, calibration_file)
    h_crit = critical_bandwidth(data, I)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resamp_fun = lambda: is_unimodal_kde(
        h_crit*lambda_alpha, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data), I)
    smaller_equal_crit_bandwidth = bootstrap(resamp_fun, N_bootstrap, dtype=np.bool_, comm=comm)
    return np.mean(~smaller_equal_crit_bandwidth)


def pval_bandwidth_fm(data, lamtol, mtol, I='auto', N_bootstrap=1000,
                      comm=MPI.COMM_WORLD):
    data = comm.bcast(data)
    I = get_I(data, I)
    lambda_alpha = 1  # TODO: Replace with correct value according to Cheng & Hall methodology
    h_crit = fisher_marron_critical_bandwidth(data, lamtol, mtol, I)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resampling_scale_factor = 1.0/np.sqrt(1+h_crit**2/np.var(data))
    smaller_equal_crit_bandwidth = bootstrap(
        is_resampled_unimodal_kde, N_bootstrap, np.bool_, comm, KDE_h_crit,
        resampling_scale_factor, len(data), h_crit*lambda_alpha, lamtol, mtol, I)
    return np.mean(~smaller_equal_crit_bandwidth)