import time
import numpy as np

from src.bandwidth_fm_test import bandwidth_test_pval_mpi

N = 1000
data = np.hstack([np.random.randn(N/2), np.random.randn(N/2)])
lamtol, mtol = 0.01, 0.01
t0 = time.time()
print "bandwidth_pval_mpi(data, lamtol, mtol) = {}".format(bandwidth_test_pval_mpi(data, lamtol, mtol, 1000))
t1 = time.time()
print "Time: {}".format(t1-t0)
