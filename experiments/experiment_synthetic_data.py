from mpi4py import MPI
import numpy as np
import sys
import os
import cPickle as pickle
import time
import traceback
from scipy.stats import binom

sys.path.append('/Users/johnsson/Forskning/Code/modality')
from src.bandwidth_fm_test import find_reference_distr
from src.bandwidth_test import pval_silverman, pval_calibrated_bandwidth
from src.util.GaussianMixture1d import GaussianMixture1d as GM
from src.calibration.lambda_alphas_calibration_bw import XSampleShoulderBW
from src.calibration.lambda_alphas_access import load_lambda
from src.util.bootstrap_MPI import bootstrap
from src.shoulder_distributions import shoulder_distribution
from src import diptest

host = 'au'

resdirs = {'ke': '/Users/johnsson/Forskning/Experiments/modality/synthetic',
           'au': '/lunarc/nobackup/users/johnsson/Results/modality/synthetic',
           'ta': '/home/johnsson/Forskning/Experiments/modality/synthetic'}


def mpiexceptabort(type, value, tb):
    traceback.print_exception(type, value, tb)
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = mpiexceptabort

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def save_res(vals, test, param):
    if rank == 0:
        resdir = resdirs[host]
        resfile = os.path.join(resdir, test+'.pkl')
        try:
            with open(resfile, 'r') as f:
                res = pickle.load(f)
        except IOError:
            res = {}
        res[param] = vals
        with open(resfile, 'w') as f:
            pickle.dump(res, f, -1)

#to_test = ['silverman', 'bandwidth_cal_normal', 'bandwidth_cal_shoulder']
to_test = ['dip_cal_normal', 'dip_cal_shoulder']  # , 'dip_nocal', 'silverman',
           #'bandwidth_cal_normal', 'bandwidth_cal_shoulder']
ntest = 500
mtol = -1
shoulder_ratio = (1, 0)
shoulder_variance = 0  # 0.25**2
I = (0, 0)
a = None

#for mtol in [0.001, 0.003, 0.0001, 0, -1]:
#for shoulder_ratio in [(2, 1), (4, 1), (6, 1), (16, 1)]:
#for shoulder_variance in [0.1**2, 0.25**2, 1]:
#for I in [(-1.5, 1.5), (-1.5, 2), (-1.5, 2.5)]:
for N in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:

    if rank == 0:

        if mtol == -1:
            np.random.seed(123+N)
            datas = [np.random.randn(N) for i in range(ntest)]
            a = None
        else:
            if shoulder_variance == 1:
                np.random.seed(126)
                if a is None:
                    a = find_reference_distr(mtol, shoulder_ratio)
                print "a = {}".format(a)
                np.random.seed(127+N+sum(shoulder_ratio)+int(abs(np.log(mtol+1e-10))))
                N_shoulders = [binom.rvs(N, shoulder_ratio[0]*1./sum(shoulder_ratio)) for i in range(ntest)]
                datas = [np.hstack([np.random.randn(N-N_shoulder), np.random.randn(N_shoulder)+a]) for i, N_shoulder in zip(range(ntest), N_shoulders)]
            else:
                weights = np.array(shoulder_ratio, dtype=np.float)
                weights /= np.sum(weights)
                if a is None:
                    if mtol == 0:
                        a, _ = shoulder_distribution(weights, np.sqrt(shoulder_variance))
                    else:
                        np.random.seed(124)
                        a = float(find_reference_distr(mtol, shoulder_ratio, shoulder_variance, min=0.2, max=4))
                print "a = {}".format(a)
                gm = GM(np.array([0, a]), np.array([1, shoulder_variance]), weights)
                np.random.seed(125+N+sum(shoulder_ratio)+int(abs(np.log(mtol+1e-10)))+int(100*np.sqrt(shoulder_variance)))
                datas = [gm.sample(N) for i in range(ntest)]

        np.random.seed()
        print "datas[0][:3] = {}".format(datas[0][:3])
    else:
        datas = None
        a = None

    datas = comm.bcast(datas)
    a = comm.bcast(a)

    if not a is None and I is None:
        I = (-1.5, a+1)

    if 'silverman' in to_test:
        t0 = time.time()
        pvals = np.zeros((ntest,), dtype=np.float)
        for i, data in enumerate(datas):
            pvals[i] = pval_silverman(data, I)
            save_res(pvals, 'silverman', (ntest, mtol, shoulder_ratio, shoulder_variance, N, I))
        t1 = time.time()
        if rank == 0:
            print "Silverman time ({}, {}, {}, {}, {}, {}): {}".format(ntest, mtol, shoulder_ratio, shoulder_variance, N, I, t1-t0)

    if 'bandwidth_cal_normal' in to_test:
        t0 = time.time()
        alpha_cal = 0.05
        #rejections = np.zeros((ntest,), dtype=np.bool)
        pvals = np.zeros((ntest,), dtype=np.float)
        for i, data in enumerate(datas):
            pvals[i] = pval_calibrated_bandwidth(data, alpha_cal, 'normal', I, comm=comm)
            #rejections[i] = reject_null_calibrated_test_bandwidth(data, alpha, 'normal', I)
        save_res(pvals, 'bandwidth_cal_normal', (ntest, mtol, shoulder_ratio, shoulder_variance, N, I, alpha_cal))
        t1 = time.time()
        if rank == 0:
            print "Calibrated normal time ({}, {}, {}, {}, {}, {}): {}".format(ntest, mtol, shoulder_ratio, shoulder_variance, N, I, t1-t0)

    if 'bandwidth_cal_shoulder' in to_test:
        t0 = time.time()
        alpha_cal = 0.05
        #rejections = np.zeros((ntest,), dtype=np.bool)
        pvals = np.zeros((ntest,), dtype=np.float)
        for i, data in enumerate(datas):
            pvals[i] = pval_calibrated_bandwidth(data, alpha_cal, 'shoulder', I, comm=comm)
            #rejections[i] = reject_null_calibrated_test_bandwidth(data, 0.05, 'shoulder', I)
        save_res(pvals, 'bandwidth_cal_shoulder', (ntest, mtol, shoulder_ratio, shoulder_variance, N, I, alpha_cal))
        t1 = time.time()
        if rank == 0:
            print "Calibrated shoulder time ({}, {}, {}, {}, {}, {}): {}".format(
                ntest, mtol, shoulder_ratio, shoulder_variance, N, I, t1-t0)

    if 'dip_nocal' in to_test:
        t0 = time.time()
        pvals = np.zeros((ntest,), dtype=np.float)
        for i, data in enumerate(datas):
            xF, yF = diptest.cum_distr(data)
            dip = diptest.dip_from_cdf(xF, yF)
            pvals[i] = diptest.dip_pval_tabinterpol(dip, len(data))
        save_res(pvals, 'dip_nocal', (ntest, mtol, shoulder_ratio, shoulder_variance, N))
        t1 = time.time()
        if rank == 0:
            print "Dip nocal time ({}, {}, {}, {}, {}): {}".format(
                ntest, mtol, shoulder_ratio, shoulder_variance, N, t1-t0)

    if 'dip_cal_shoulder' in to_test or 'dip_cal_normal' in to_test:
        for null in ['shoulder', 'normal']:
            if not 'dip_cal_'+null in to_test:
                continue
            alpha_cal = 0.05
            N_bootstrap = 1000
            lambda_alpha = load_lambda('dip_ex', null, alpha_cal)
            if rank == 0:
                print "lambda_alpha = {}".format(lambda_alpha)
            t0 = time.time()
            pvals = np.zeros((ntest,), dtype=np.float)
            for i, data in enumerate(datas):
                xF, yF = diptest.cum_distr(data)
                dip, unimod = diptest.dip_and_closest_unimodal_from_cdf(xF, yF)
                resamp_fun = lambda: diptest.dip_resampled_from_unimod(unimod, len(data))
                resamp_dips = bootstrap(resamp_fun, N_bootstrap, dtype=np.float_, comm=comm)
                pvals[i] = np.mean(resamp_dips > lambda_alpha*dip)
            save_res(pvals, 'dip_cal_{}'.format(null), (ntest, mtol, shoulder_ratio, shoulder_variance, N, alpha_cal))
            t1 = time.time()
            if rank == 0:
                print "Dip cal {} time ({}, {}, {}, {}, {}): {}".format(
                    null, ntest, mtol, shoulder_ratio, shoulder_variance, N, t1-t0)

    if 'bandwidth_cal_shoulder_ref' in to_test:
        if shoulder_ratio == (16, 1) and shoulder_variance == 0.25**2 and mtol == 0 and I == (-1.5, 1.5):
            alpha_cal = 0.05
            N_bootstrap = 1000
            lambda_alpha = load_lambda('bw', 'shoulder', alpha_cal)
            print "lambda_alpha = {}".format(lambda_alpha)
            t0 = time.time()
            pvals = np.zeros((ntest,), dtype=np.float)
            for i, data in enumerate(datas):
                xsamp = XSampleShoulderBW(N, comm=comm)
                resamp_fun = lambda: xsamp.is_unimodal_resample(lambda_alpha)
                smaller_equal_crit_bandwidth = bootstrap(resamp_fun, N_bootstrap, dtype=np.bool_, comm=xsamp.comm)
                pvals[i] = np.mean(~smaller_equal_crit_bandwidth)
                #rejections[i] = reject_null_calibrated_test_bandwidth(data, 0.05, 'shoulder', I)
            save_res(pvals, 'bandwidth_cal_shoulder_ref', (ntest, mtol, shoulder_ratio, shoulder_variance, N, I, alpha_cal))
            t1 = time.time()
            if rank == 0:
                print "Calibrated shoulder ref time ({}, {}, {}, {}, {}, {}): {}".format(ntest, mtol, shoulder_ratio, shoulder_variance, N, I, t1-t0)

    if 'bandwidth_cal_shoulder_ref2' in to_test:
        if shoulder_ratio == (16, 1) and shoulder_variance == 0.25**2 and mtol == 0 and I == (-1.5, 1.5):
            alpha_cal = 0.05
            N_bootstrap = 1000
            lambda_alpha = load_lambda('bw', 'shoulder', alpha_cal)
            print "lambda_alpha = {}".format(lambda_alpha)
            t0 = time.time()
            pvals = np.zeros((ntest,), dtype=np.float)
            for i, data in enumerate(datas):
                xsamp = XSampleShoulderBW(N, comm=comm)
                pvals[i] = pval_calibrated_bandwidth(xsamp.data, alpha_cal, 'shoulder', I, comm=xsamp.comm)
                #rejections[i] = reject_null_calibrated_test_bandwidth(data, 0.05, 'shoulder', I)
            save_res(pvals, 'bandwidth_cal_shoulder_ref2', (ntest, mtol, shoulder_ratio, shoulder_variance, N, I, alpha_cal))
            t1 = time.time()
            if rank == 0:
                print "Calibrated shoulder ref time ({}, {}, {}, {}, {}, {}): {}".format(ntest, mtol, shoulder_ratio, shoulder_variance, N, I, t1-t0)



