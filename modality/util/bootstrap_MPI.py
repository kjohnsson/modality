from __future__ import unicode_literals
from __future__ import print_function
from mpi4py import MPI
import numpy as np
from scipy.stats import binom

from . import print_rank0, print_all_ranks

#comm = MPI.COMM_WORLD

# def bootstrap(fun, N, dtype=np.float_, *args):
#     res = np.zeros((N,), dtype=dtype)
#     for i in range(len(res)):
#         res[i] = fun(*args)
#     return res


class MaxSampExceededException(Exception):
    pass


def check_equal_mpi(comm, val):
    rank = comm.Get_rank()
    val_hash = hash(str(val))
    val_hash_rank0 = comm.bcast(val_hash)
    not_same_data = val_hash != val_hash_rank0

    not_same_data_all = comm.gather(not_same_data)
    if rank == 0:
        not_same_data_any = False
        for ns in not_same_data_all:
            if ns:
                not_same_data_any = True
    else:
        not_same_data_any = False
    not_same_data_any = comm.bcast(not_same_data_any)
    if not_same_data_any:
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


def probability_in_interval(fun_resample, gamma_lower, gamma_upper,
                            significance_first=0.01, significance_second=0.05,
                            batch=5, comm=MPI.COMM_SELF,
                            print_per_batch=False, printing=True):
    N_test_max = 20000
    vals = np.zeros((0,))
    s = "gamma_lower, gamma_upper = {}, {}".format(gamma_lower, gamma_upper)
    while True:
        vals_new_samp = bootstrap(fun_resample, batch, comm=comm)
        vals = np.hstack([vals, vals_new_samp])
        upper_bound_pval = binom.cdf(np.sum(vals), len(vals), gamma_upper)
        lower_bound_pval = 1 - binom.cdf(np.sum(vals)-1, len(vals), gamma_lower)
        s += ("\nnp.mean(vals) = {}".format(np.mean(vals)) +
              "\nlen(vals) = {}".format(len(vals)) +
              "\nupper_bound_pval = {}".format(upper_bound_pval) +
              "\nlower_bound_pval = {}".format(lower_bound_pval))
        if upper_bound_pval < significance_first:
            if lower_bound_pval < significance_second:
                s += '\n===\nin interval\n==='
                print_rank0(comm, s)
                return 'in interval'
            if 1-binom.cdf(int(np.round(np.mean(vals)*N_test_max))-1, N_test_max, gamma_lower) < significance_second:
                batch = len(vals)
                continue  # Expecting less than N_test_max tests to verify lower bound
            s += '\n===\nbelow upper bound\n==='
            if printing:
                print_rank0(comm, s)
            return 'below upper bound'
        if lower_bound_pval < significance_first:
            if upper_bound_pval < significance_second:
                s += '\n===\nin interval\n==='
                if printing:
                    print_rank0(comm, s)
                return 'in interval'
            if binom.cdf(int(np.round(np.mean(vals)*N_test_max)), N_test_max, gamma_upper) < significance_second:
                batch = len(vals)
                continue  # Expecting less than N_test_max tests to verify upper bound
            s += '\n===\nabove lower bound\n==='
            if printing:
                print_rank0(comm, s)
            return 'above lower bound'
        batch = len(vals)
        if print_per_batch:
            if printing:
                print_rank0(comm, s)
            s = "gamma_lower, gamma_upper = {}, {}".format(gamma_lower, gamma_upper)


def probability_above(fun_resample, gamma, max_samp=None, comm=MPI.COMM_SELF,
                      batch=5, tol=0, bound_significance=0.01, print_per_batch=False,
                      exception_at_max_samp=False, printing=True):
    '''
        Returns True if P(fun_resample()) is significantly above gamma,
        returns False if P(fun_resample()) is significantly below gamma.
        Increases samples size until significance is obtained.
        (null hypothesis is p = gamma).
    '''
    vals = np.zeros((0,))
    s = "gamma = {}".format(gamma)
    while True:
        vals_new_samp = bootstrap(fun_resample, batch, comm=comm, dtype=np.bool_)
        #if True:#gamma == 0.05:
        #    print_all_ranks(comm, str(vals_new_samp))
        #vals_new_samp = vals_new_samp[~np.isnan(vals_new_samp)]
        vals = np.hstack([vals, vals_new_samp.astype(np.float_)])
        upper_bound_pval = binom.cdf(np.sum(vals), len(vals), gamma+tol)
        lower_bound_pval = 1 - binom.cdf(np.sum(vals)-1, len(vals), gamma-tol)

        s += ("\nnp.mean(vals) = {}".format(np.mean(vals)) +
              "\nlen(vals) = {}".format(len(vals)) +
              "\nupper_bound_pval = {}".format(upper_bound_pval) +
              "\nlower_bound_pval = {}".format(lower_bound_pval))
        if upper_bound_pval <= bound_significance:
            s += '\n---'
            if printing:
                print_rank0(comm, s)
            return False  # we have lower bound instead.
        if lower_bound_pval <= bound_significance:
            s += "\n---"
            if printing:
                print_rank0(comm, s)
            return True
        if not max_samp is None:
            if len(vals) > max_samp:
                if exception_at_max_samp:
                    raise MaxSampExceededException
                s += "\n---"+"\n"+"max_samp reached"
                print_rank0(comm, s)
                lower_bound = np.random.rand(1) < 0.5
                lower_bound = comm.bcast(lower_bound)
                if lower_bound:  # 50% chance to be above or below
                    return True
                return False
        batch = len(vals)
        if print_per_batch:
            if printing:
                print_rank0(comm, s)
            s = "gamma = {}".format(gamma)


if __name__ == '__main__':
    if 0:
        def testfun(arg1, arg2):
            return 0.01*np.abs(np.random.randn(1))+arg1+arg2

        N = 5
        print("bootstrap(testfun, N, rank, size) = {}".format(bootstrap(testfun, N, np.float_, MPI.COMM_WORLD, rank, size)))

    def testfun():
        alpha = 0.08
        return np.random.rand() < alpha

    print("probability_in_interval(testfun, 0.07, 0.09) = {}".format(probability_in_interval(testfun, 0.07, 0.09)))
