import numpy as np
from mpi4py import MPI
from scipy.stats import binom

from . import print_rank0, print_all_ranks

#comm = MPI.COMM_WORLD

# def bootstrap(fun, N, dtype=np.float_, *args):
#     res = np.zeros((N,), dtype=dtype)
#     for i in range(len(res)):
#         res[i] = fun(*args)
#     return res


def check_equal_mpi(comm, val):
    val_hash = hash(str(val))
    val_hash_rank0 = comm.bcast(val_hash)
    if val_hash != val_hash_rank0:
        raise ValueError('Not same data across workers.')


def bootstrap(fun, N, dtype=np.float_, comm=MPI.COMM_SELF, *args):
    rank = comm.Get_rank()
    size = comm.Get_size()
    #seed = np.random.randint(100000)+rank
    #np.random.seed(seed)  # ensure different random numbers at different workers
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


def bootstrap_array(fun, N, l, dtype=np.float_, *args):
    res = np.zeros((N, l), dtype=dtype)
    for i in range(res.shape[0]):
        res[i, :] = fun(*args)
    return res


def probability_above(fun_resample, gamma, max_samp=None, comm=MPI.COMM_SELF,
                      batch=5, tol=0, bound_significance=0.01, print_per_batch=False):
    '''
        Returns True if P(fun_resample()) is significantly above gamma,
        returns False if P(fun_resample()) is significantly below gamma.
        Increases samples size until significance is obtained.
        (null hypothesis is p = gamma).
    '''
    vals = np.zeros((0,))
    s = "gamma = {}".format(gamma)
    while True:
        vals_new_samp = bootstrap(fun_resample, batch, comm=comm)
        if gamma == 0.05:
            print_all_ranks(comm, str(vals_new_samp))
        #vals_new_samp = vals_new_samp[~np.isnan(vals_new_samp)]
        vals = np.hstack([vals, vals_new_samp])
        upper_bound_pval = binom.cdf(np.sum(vals), len(vals), gamma+tol)
        lower_bound_pval = 1 - binom.cdf(np.sum(vals)-1, len(vals), gamma-tol)

        s += ("\nnp.mean(vals) = {}".format(np.mean(vals)) +
              "\nlen(vals) = {}".format(len(vals)) +
              "\nupper_bound_pval = {}".format(upper_bound_pval) +
              "\nlower_bound_pval = {}".format(lower_bound_pval))
        if upper_bound_pval <= bound_significance:
            s += '\n---'
            print_rank0(comm, s)
            return False  # we have lower bound instead.
        if lower_bound_pval <= bound_significance:
            s += "\n---"
            print_rank0(comm, s)
            return True
        if not max_samp is None:
            if len(vals) > max_samp:
                s += "\n---"+"\n"+"max_samp reached"
                print_rank0(comm, s)
                lower_bound = np.random.rand(1) < 0.5
                lower_bound = comm.bcast(lower_bound)
                if lower_bound:  # 50% chance to be above or below
                    return True
                return False
        batch = len(vals)
        if print_per_batch:
            print_rank0(comm, s)
            s = "gamma = {}".format(gamma)


if __name__ == '__main__':
    def testfun(arg1, arg2):
        return 0.01*np.abs(np.random.randn(1))+arg1+arg2

    N = 5
    print "bootstrap_mpi(testfun, N, rank, size) = {}".format(bootstrap_mpi(testfun, N, rank, size))