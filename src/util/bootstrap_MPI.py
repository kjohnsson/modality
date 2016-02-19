import numpy as np
from mpi4py import MPI
from scipy.stats import binom

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def bootstrap(fun, N, dtype=np.float_, *args):
    res = np.zeros((N,), dtype=dtype)
    for i in range(len(res)):
        res[i] = fun(*args)
    return res


def bootstrap_mpi(fun, N, dtype=np.float_, *args):
    seed = np.random.randint(1000)+rank
    np.random.seed(seed)
    res_loc = np.array_split(np.zeros((N,), dtype=dtype), size)
    res_loc = comm.scatter(res_loc)
    args = comm.bcast(args)
    for i in range(len(res_loc)):
        res_loc[i] = fun(*args)
    res_loc = comm.gather(res_loc)
    if rank == 0:
        res = np.hstack(res_loc)
    else:
        res = None
    res = comm.bcast(res)
    return res


def expected_value_above(fun_resample, gamma, max_samp=None, mpi=False):
    batch = 20
    bound_significance = 0.01
    vals = np.zeros((0,))
    while True:
        if not mpi:
            vals_new_samp = bootstrap(fun_resample, batch)
        else:
            vals_new_samp = bootstrap_mpi(fun_resample, batch)
        vals_new_samp = vals_new_samp[~np.isnan(vals_new_samp)]
        vals = np.hstack([vals, vals_new_samp])
        p = np.mean(vals)
        bound_pval = binom.cdf(gamma*len(vals), len(vals), p)
        if rank == 0:
            print ("gamma = {}".format(gamma)+"\nnp.mean(vals) = {}".format(np.mean(vals)) +
                   "\nlen(vals) = {}".format(len(vals))+"\nbound_pval = {}".format(bound_pval))
        if bound_pval < bound_significance:
            if rank == 0:
                print "---"
            return True
        if bound_pval > 1-bound_significance:
            if rank == 0:
                print "---"
            return False  # we have lower bound instead.
        if not max_samp is None:
            if len(vals) > max_samp:
                if rank == 0:
                    print "---"
                return np.nan
        batch = len(vals)


if __name__ == '__main__':
    def testfun(arg1, arg2):
        return 0.01*np.abs(np.random.randn(1))+arg1+arg2

    N = 5
    print "bootstrap_mpi(testfun, N, rank, size) = {}".format(bootstrap_mpi(testfun, N, rank, size))