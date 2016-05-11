from mpi4py import MPI
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

from ..diptest import cum_distr, dip_and_closest_unimodal_from_cdf, dip_from_cdf, sample_from_unimod
from ..util.bootstrap_MPI import bootstrap, bootstrap_array, probability_above
from .lambda_alphas_access import save_lambda
from ..util import print_rank0, print_all_ranks


def normalsamp(N, comm):
    if comm.Get_rank() == 0:
        data = np.random.randn(N)
    else:
        data = None
    data = comm.bcast(data)
    return data


def shouldersamp(N, comm):
    if comm.Get_rank() == 0:
        N1 = binom.rvs(N, 1.0/17)
        N2 = N - N1
        m1 = -1.25
        s1 = 0.25
        data = np.hstack([s1*np.random.randn(N1)+m1, np.random.randn(N2)])
    else:
        data = None
    data = comm.bcast(data)
    return data


class XSampleDip(object):

    def __init__(self, N, sampfun, comm=MPI.COMM_WORLD):
        self.N = N
        self.comm = comm
        self.rank = self.comm.Get_rank()
        data = sampfun(N, self.comm)
        xF, yF = cum_distr(data)
        self.dip, self.unimod = dip_and_closest_unimodal_from_cdf(xF, yF)

    def dip_resampled(self):
        data = self.sample_from_unimod()
        xF, yF = cum_distr(data)
        return dip_from_cdf(xF, yF)

    def sample_from_unimod(self):
        return sample_from_unimod(self.unimod, self.N)

    def lowest_lambdas_rejecting(self, alphas, B=1000):
        dips = bootstrap(self.dip_resampled, B, comm=self.comm)
        i_s = np.floor(alphas*B)
        dip_thrs = -np.sort(-dips)[i_s.astype(np.int)]
        lambdas = dip_thrs/self.dip
        #print "np.mean(dips/self.dip <= lambd) = {}".format(np.mean(dips/self.dip <= lambd))
        return lambdas

    def probability_of_unimodal_above(self, lambda_val, gamma):
        '''
            G_n(\lambda) = P(\hat Delta_{crit}^*/\hat Delta_{crit} <= \lambda)
        '''
        return probability_above(lambda: self.dip_resampled()/self.dip < lambda_val,
                                 gamma, max_samp=5000, comm=self.comm, batch=20)

    def plot_unimodal(self):
        plt.plot(*self.unimod)


def dip_scale_factor(alpha, null='normal', lower_lambda=0, upper_lambda=2.0,
                     comm=MPI.COMM_WORLD, **samp_class_args):

    rank = comm.Get_rank()
    sampfun = normalsamp if null == 'normal' else shouldersamp

    def print_bound_search(fun):

        def printfun(lambda_val):
            print_rank0(comm, "Testing if {} is upper bound for lambda_alpha".format(lambda_val))
            res = fun(lambda_val)
            print_rank0(comm, "{} is".format(lambda_val)+" not"*(not res)+" upper bound for lambda_alpha.")
            return res

        return printfun

    @print_bound_search
    def is_upper_bound_on_lambda(lambda_val):
        '''
            P(P(G_n(lambda)) > 1 - alpha) > alpha
                => lambda is upper bound on lambda_alpha
        '''
        return probability_above(
            lambda: XSampleDip(N, sampfun, comm=comm).probability_of_unimodal_above(
                lambda_val, 1-alpha), alpha, comm=MPI.COMM_SELF, batch=20, tol=0.005, print_per_batch=True)  # 0.005)

    def save_upper(lambda_bound):
        save_lambda(lambda_bound, 'dip_ex', null, alpha, upper=True)

    def save_lower(lambda_bound):
        save_lambda(lambda_bound, 'dip_ex', null, alpha, upper=False)

    lambda_tol = 1e-4

    N = 10000
    seed = np.random.randint(1000)
    seed = comm.bcast(seed)
    seed += rank
    #seed = 846
    #seeds = [1013, 225, 603, 112, 952, 870, 869, 394, 986, 458, 685, 438, 74,
    #         671, 356, 255, 241, 802, 339, 193]
    #sseed = seeds[MPI.COMM_WORLD.Get_rank()]
    print_all_ranks(comm, "seed = {}".format(seed))
    np.random.seed(seed)

    if lower_lambda == 0:
        new_lambda = upper_lambda/2
        while is_upper_bound_on_lambda(new_lambda):
            upper_lambda = new_lambda
            save_upper(upper_lambda)
            new_lambda = (upper_lambda+lower_lambda)/2
        lower_lambda = new_lambda
        save_lower(lower_lambda)

    while upper_lambda-lower_lambda > lambda_tol:
        new_lambda = (upper_lambda+lower_lambda)/2
        if is_upper_bound_on_lambda(new_lambda):
            upper_lambda = new_lambda
            save_upper(upper_lambda)
        else:
            lower_lambda = new_lambda
            save_lower(lower_lambda)

    return (upper_lambda+lower_lambda)/2


def dip_scale_factor_approx(alphas, sampfun, B=1000, N=10000):
    i_s = np.round(B*alphas)-1
    lowest_lambdas_rejects = bootstrap_array(lambda: XSampleDip(N, sampfun).lowest_lambdas_rejecting(alphas), B, len(alphas))
    lambdas = np.sort(lowest_lambdas_rejects, axis=0)[i_s.astype(np.int), np.arange(len(alphas))]
    #print "np.mean(lowest_lambdas_rejecting <= lambd) = {}".format(np.mean(lowest_lambdas_rejecting <= lambd))
    return lambdas


if __name__ == '__main__':
    #seed = np.random.randint(1000)
    import time
    seed = 123  # 411
    rank = MPI.COMM_WORLD.Get_rank()
    print "seed = {} at rank {}".format(seed+rank, rank)
    np.random.seed(seed+rank)
    if 0:
        xdip = XSampleDip(1000, normalsamp)
        if 0:
            data = xdip.sample_from_unimod()
            xF, yF = cum_distr(data)
            xdip.plot_unimodal()
            plt.plot(xF, yF)
            plt.show()
        if 0:
            print "xdip.dip = {}".format(xdip.dip)
            for i in range(10):
                print "xdip.dip_resampled() = {}".format(xdip.dip_resampled())
        if 0:
            print "xdip.lowest_lambda_rejecting(0.05) = {}".format(xdip.lowest_lambda_rejecting(0.05))
    #print "dip_scale_factor(0.05) = {}".format(dip_scale_factor(0.05))
    alphas = np.arange(0.01, 0.99, 0.01)
    t0 = time.time()
    lambda_alphas = dip_scale_factor(alphas, normalsamp)
    t1 = time.time()
    print "Time: {}".format(t1-t0)
    for alpha, lambda_alpha in zip(alphas, lambda_alphas):
        save_lambda(lambda_alpha, 'dip', 'normal', alpha)
    #plt.plot(alphas, lambda_alphas)
    #plt.show()
