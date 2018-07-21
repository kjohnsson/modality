from __future__ import unicode_literals
'''
    Alternative calibration of dip test (no resampling) based on
        M.-Y. Cheng and P. Hall. 1999: Calibrating the excess mass and
        dip tests of modality. Journal of the Royal Statistical Society:
        Series B (Statistical Methodology), 60(3): 579–589, 1998.
'''
from __future__ import print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta as betafun
from scipy.optimize import brentq

from diptest import cum_distr, dip_from_cdf, dip_pval_tabinterpol
from KernelDensityDerivative import KernelDensityDerivative


def calibrated_dip_test(data, N_bootstrap=1000):
    xF, yF = cum_distr(data)
    dip = dip_from_cdf(xF, yF)
    n_eval = 512
    f_hat = KernelDensityDerivative(data, 0)
    f_bis_hat = KernelDensityDerivative(data, 2)
    x = np.linspace(np.min(data), np.max(data), n_eval)
    f_hat_eval = f_hat.evaluate(x)
    ind_x0_hat = np.argmax(f_hat_eval)
    d_hat = np.abs(f_bis_hat.evaluate(x[ind_x0_hat]))/f_hat_eval[ind_x0_hat]**3
    ref_distr = select_calibration_distribution(d_hat)
    ref_dips = np.zeros(N_bootstrap)
    for i in range(N_bootstrap):
        samp = ref_distr.sample(len(data))
        xF, yF = cum_distr(samp)
        ref_dips[i] = dip_from_cdf(xF, yF)
    return np.mean(ref_dips > dip)


def select_calibration_distribution(d_hat):
    data_dir = os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'data')
    with open(os.path.join(data_dir, 'gammaval.pkl'), 'r') as f:
        savedat = pickle.load(f)

    if np.abs(d_hat-np.pi) < 1e-4:
        return RefGaussian()
    if d_hat < 2*np.pi:  # beta distribution
        gamma = lambda beta: 2*(beta-1)*betafun(beta, 1.0/2)**2 - d_hat
        i = np.searchsorted(savedat['gamma_betadistr'], d_hat)
        beta_left = savedat['beta_betadistr'][i-1]
        beta_right = savedat['beta_betadistr'][i]
        beta = brentq(gamma, beta_left, beta_right)
        return RefBeta(beta)

    # student t distribution
    gamma = lambda beta: 2*beta*betafun(beta-1./2, 1./2)**2 - d_hat
    i = np.searchsorted(-savedat['gamma_studentt'], -d_hat)
    beta_left = savedat['beta_studentt'][i-1]
    beta_right = savedat['beta_studentt'][i]
    beta = brentq(gamma, beta_left, beta_right)
    return RefStudentt(beta)


class RefGaussian(object):
    def sample(self, n):
        return np.random.randn(n)


class RefBeta(object):
    def __init__(self, beta):
        self.beta = beta

    def sample(self, n):
        return np.random.beta(self.beta, self.beta, n)


class RefStudentt(object):
    def __init__(self, beta):
        self.beta = beta

    def sample(self, n):
        dof = 2*self.beta-1
        return 1./np.sqrt(dof)*np.random.standard_t(dof, n)

if __name__ == '__main__':

    if 0:
        ntest = 100
        pvals_calib = np.zeros(ntest)
        pvals_direct = np.zeros(ntest)
        for i in range(ntest):
            data = np.random.randn(100)
            pvals_calib[i] = calibrated_dip_test(data)
            xF, yF = cum_distr(data)
            dip = dip_from_cdf(xF, yF)
            pvals_direct[i] = dip_pval_tabinterpol(dip, len(data))

        from beeswarm import beeswarm
        beeswarm([pvals_calib, pvals_direct])  # should be uniform distribution between 0 and 1
        plt.show()

    if 1:
        import time
        np.random.seed(1)
        data = np.random.randn(100)
        np.random.seed()
        t0 = time.time()
        pval = calibrated_dip_test(data)
        t1 = time.time()
        print("pval = {}".format(pval))
        print("Time consumption calibrated dip test: {}".format(t1-t0))
