from __future__ import unicode_literals
from __future__ import print_function
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from .XSample import XSample
from .reference_sampfun import normalsamp, shouldersamp
from ..diptest import cum_distr, dip_and_closest_unimodal_from_cdf, dip_from_cdf, sample_from_unimod
from ..util.bootstrap_MPI import bootstrap, bootstrap_array, probability_above
from .lambda_alphas_access import save_lambda
from ..util import print_rank0, print_all_ranks, fp_blurring


class XSampleDip(XSample):
    '''
        Class that samples a data set from a reference distribution,
        computes dip and closest unimodal distribution and from which
        data from the closest unimodal distribution can be sampled.
    '''

    def __init__(self, N, sampfun, comm=MPI.COMM_WORLD):
        super(XSampleDip, self).__init__(N, sampfun, comm)

    @property
    def statistic(self):
        return self.dip

    @property
    def dip(self):
        try:
            return self._dip
        except AttributeError:
            self.compute_dip()
            return self._dip

    @property
    def unimod(self):
        try:
            return self._unimod
        except AttributeError:
            self.compute_dip()
            return self._unimod

    def compute_dip(self):
        xF, yF = cum_distr(self.data)
        self._dip, self._unimod = dip_and_closest_unimodal_from_cdf(xF, yF)

    def resampled_statistic_below_scaled_statistic(self, lambda_scale):
        return self.dip_resampled() < lambda_scale*self.dip

    def dip_resampled(self):
        data = self.sample_from_unimod()
        xF, yF = cum_distr(data)
        return dip_from_cdf(xF, yF)

    def sample_from_unimod(self):
        return sample_from_unimod(self.unimod, self.N)

    def lowest_lambdas_rejecting(self, alphas, B=1000):
        '''
            Returning the lowest lambdas rejecting the null hypothesis
            of unimodality at significance levels alphas.

            B   -   number of bootstrap samples
        '''
        dips = bootstrap(self.dip_resampled, B, comm=self.comm)
        i_s = np.floor(alphas*B)
        dip_thrs = -np.sort(-dips)[i_s.astype(np.int)]
        lambdas = dip_thrs/self.dip
        #print "np.mean(dips/self.dip <= lambd) = {}".format(np.mean(dips/self.dip <= lambd))
        return lambdas

    def prob_resampled_dip_below_bound_above_gamma(self, lambda_scale, gamma):
        '''
            Is the probability that a resampled dip is below
            lambda_val*(original dip), significantly above gamma
            (returns True) or significantly below gamma (returns False)?
            Significance level of bound is 0.01. If after 5000 samples
            this cannot be determined, the result True/False is drawn
            by random with equal probabilities 0.5.

            Equivalently, with
            G_n(\lambda) = P(Delta^*/Delta <= \lambda)
            is G_n(\lambda) significantly above or significantly below gamma?
        '''
        return self.prob_resampled_statistic_below_bound_above_gamma(lambda_scale, gamma)

    def plot_unimodal(self):
        plt.plot(*self.unimod)


class XSampleDipTrunc(XSampleDip):

    def __init__(self, N, sampfun, range_, comm=MPI.COMM_WORLD, blur_func=None):
        super(XSampleDipTrunc, self).__init__(N, sampfun, comm)
        #self.data = self.data[(self.data > -3) & (self.data < 3)]
        #print "nbr removed: {}".format(N-len(self.data))

        if blur_func is None:
            blur_func = lambda x: x
        self.blur_func = blur_func
        self.range_ = range_

    def set_data(self, data):
        self.data = np.round((data+3)*self.range_/6)
        self.data = self.blur_func(self.data)
        self.compute_dip()

    # def sample_from_unimod(self):
    #     data = sample_from_unimod(self.unimod, self.N)
    #     return self.blur_func(np.round(data))


def dip_scale_factor(alpha, null='normal', lower_lambda=0, upper_lambda=2.0,
                     comm=MPI.COMM_WORLD, save_file=None):

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
            lambda: XSampleDip(N, sampfun, comm=comm).prob_resampled_dip_below_bound_above_gamma(
                lambda_val, 1-alpha), alpha, comm=MPI.COMM_SELF, batch=20, tol=0, print_per_batch=True)  # 0.005)

    def save_upper(lambda_bound):
        save_lambda(lambda_bound, 'dip_ex', null, alpha, upper=True, lambda_file=save_file)

    def save_lower(lambda_bound):
        save_lambda(lambda_bound, 'dip_ex', null, alpha, upper=False, lambda_file=save_file)

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
    print("seed = {} at rank {}".format(seed+rank, rank))
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
            print("xdip.dip = {}".format(xdip.dip))
            for i in range(10):
                print("xdip.dip_resampled() = {}".format(xdip.dip_resampled()))
        if 0:
            print("xdip.lowest_lambda_rejecting(0.05) = {}".format(xdip.lowest_lambda_rejecting(0.05)))
    #print "dip_scale_factor(0.05) = {}".format(dip_scale_factor(0.05))
    alphas = np.arange(0.01, 0.99, 0.01)
    t0 = time.time()
    lambda_alphas = dip_scale_factor(alphas, normalsamp)
    t1 = time.time()
    print("Time: {}".format(t1-t0))
    for alpha, lambda_alpha in zip(alphas, lambda_alphas):
        save_lambda(lambda_alpha, 'dip', 'normal', alpha)
    #plt.plot(alphas, lambda_alphas)
    #plt.show()
