from mpi4py import MPI
import numpy as np
from sklearn.neighbors import KernelDensity

from .util.bootstrap_MPI import bootstrap, check_equal_mpi
from .calibration.lambda_alphas_access import load_lambda
from . import diptest
from .critical_bandwidth import critical_bandwidth, is_unimodal_kde
from .critical_bandwidth_fm import fisher_marron_critical_bandwidth, is_resampled_unimodal_kde


def pval_calibrated_dip(data, alpha_cal, null, N_bootstrap=1000, comm=MPI.COMM_WORLD):
    '''
        NB!: Test is only calibrated to correct level for alpha_cal.
    '''
    check_equal_mpi(comm, data)
    lambda_alpha = load_lambda('dip_ex', null, alpha_cal)
    xF, yF = diptest.cum_distr(data)
    dip, unimod = diptest.dip_and_closest_unimodal_from_cdf(xF, yF)
    resamp_fun = lambda: diptest.dip_resampled_from_unimod(unimod, len(data))
    resamp_dips = bootstrap(resamp_fun, N_bootstrap, dtype=np.float_, comm=comm)
    return np.mean(resamp_dips > lambda_alpha*dip)


def pval_silverman(data, I=(-np.inf, np.inf), N_bootstrap=1000, comm=MPI.COMM_WORLD):
    check_equal_mpi(comm, data)
    h_crit = critical_bandwidth(data, I)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resamp_fun = lambda: is_unimodal_kde(
        h_crit, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data), I)
    smaller_equal_crit_bandwidth = bootstrap(resamp_fun, N_bootstrap, dtype=np.bool_,
                                             comm=comm)
    return np.mean(~smaller_equal_crit_bandwidth)


def pval_calibrated_bandwidth(data, alpha_cal, null, I=(-np.inf, np.inf),
                              N_bootstrap=1000, comm=MPI.COMM_WORLD):
    '''
        NB!: Test is only calibrated to correct level for alpha_cal.
    '''
    check_equal_mpi(comm, data)
    lambda_alpha = load_lambda('bw', null, alpha_cal)
    h_crit = critical_bandwidth(data, I)
    var_data = np.var(data)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resamp_fun = lambda: is_unimodal_kde(
        h_crit*lambda_alpha, KDE_h_crit.sample(len(data)).ravel()/np.sqrt(1+h_crit**2/var_data), I)
    smaller_equal_crit_bandwidth = bootstrap(resamp_fun, N_bootstrap, dtype=np.bool_, comm=comm)
    return np.mean(~smaller_equal_crit_bandwidth)


def pval_bandwidth_fm(data, lamtol, mtol, I=(-np.inf, np.inf), N_bootstrap=1000,
                      comm=MPI.COMM_WORLD):
    check_equal_mpi(comm, data)
    lambda_alpha = 1  # TODO: Replace with correct value according to Cheng & Hall methodology
    h_crit = fisher_marron_critical_bandwidth(data, lamtol, mtol, I)
    KDE_h_crit = KernelDensity(kernel='gaussian', bandwidth=h_crit).fit(data.reshape(-1, 1))
    resampling_scale_factor = 1.0/np.sqrt(1+h_crit**2/np.var(data))
    smaller_equal_crit_bandwidth = bootstrap(
        is_resampled_unimodal_kde, N_bootstrap, np.bool_, comm, KDE_h_crit,
        resampling_scale_factor, len(data), h_crit*lambda_alpha, lamtol, mtol, I)
    return np.mean(~smaller_equal_crit_bandwidth)
