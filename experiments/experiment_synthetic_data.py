import numpy as np
import sys
import os
import cPickle as pickle
import time
from mpi4py import MPI
import traceback
from scipy.stats import binom

sys.path.append('/home/johnsson/Forskning/Code/modality')
from src.bandwidth_fm_test import find_reference_distr
from src.bandwidth_test import pval_silverman, reject_null_calibrated_test_bandwidth

host = 'au'

resdirs = {'ke': '/Users/johnsson/Forskning/Experiments/modality/synthetic',
           'au': '/lunarc/nobackup/users/johnsson/Results/modality/synthetic',
           'ta': '/home/johnsson/Forskning/Experiments/modality/synthetic'}


def mpiexceptabort(type, value, tb):
    traceback.print_exception(type, value, tb)
    MPI.COMM_WORLD.Abort(1)

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

to_test = ['silverman', 'bandwidth_cal_normal', 'bandwidth_cal_shoulder']
ntest = 100
mtol = 0

#for mtol in [-1, 0, 0.001, 0.003, 0.0001]:
for shoulder_ratio in [(1, 3), (1, 5), (1, 7), (1, 17)]:
    for N in [1000, 2000, 5000, 10000]:

        if rank == 0:
            np.random.seed(123)
            if mtol == -1:
                datas = [np.random.randn(N) for i in range(ntest)]
            else:
                a = find_reference_distr(mtol, shoulder_ratio)
                print "a = {}".format(a)
                N_shoulders = [binom.rvs(N, shoulder_ratio[0]*1./shoulder_ratio[1]) for i in range(ntest)]
                datas = [np.hstack([np.random.randn(N-N_shoulder), np.random.randn(N_shoulder)+a]) for i, N_shoulder in zip(range(ntest), N_shoulders)]
            np.random.seed()
            print "datas[0][:3] = {}".format(datas[0][:3])
        else:
            datas = None
            a = None

        datas = comm.bcast(datas)
        a = comm.bcast(a)

        I = (-1.5, a+1.5)

        if 'silverman' in to_test:
            t0 = time.time()
            pvals = np.zeros((ntest,), dtype=np.float)
            for i, data in enumerate(datas):
                pvals[i] = pval_silverman(data, I)
                save_res(pvals, 'silverman', (ntest, mtol, shoulder_ratio, N))
            t1 = time.time()
            if rank == 0:
                print "Silverman time ({}, {}, {}, {}): {}".format(ntest, mtol, shoulder_ratio, N, t1-t0)

        if 'bandwidth_cal_normal' in to_test:
            t0 = time.time()
            alpha = 0.05
            rejections = np.zeros((ntest,), dtype=np.bool)
            for i, data in enumerate(datas):
                rejections[i] = reject_null_calibrated_test_bandwidth(data, alpha, 'normal', I)
            save_res(rejections, 'bandwidth_cal_normal', (ntest, mtol, shoulder_ratio, N, alpha))
            t1 = time.time()
            if rank == 0:
                print "Calibrated normal time ({}, {}, {}, {}): {}".format(ntest, mtol, shoulder_ratio, N, t1-t0)

        if 'bandwidth_cal_shoulder' in to_test:
            t0 = time.time()
            alpha = 0.05
            rejections = np.zeros((ntest,), dtype=np.bool)
            for i, data in enumerate(datas):
                rejections[i] = reject_null_calibrated_test_bandwidth(data, 0.05, 'shoulder', I)
            save_res(rejections, 'bandwidth_cal_shoulder', (ntest, mtol, shoulder_ratio, N, alpha))
            t1 = time.time()
            if rank == 0:
                print "Calibrated shoulder time ({}, {}, {}, {}): {}".format(ntest, mtol, shoulder_ratio, N, t1-t0)
