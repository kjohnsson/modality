from __future__ import unicode_literals
from __future__ import print_function

import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import binom

from .XSample import XSample
from .lambda_alphas_access import save_lambda
from ..critical_bandwidth import is_unimodal_kde, critical_bandwidth
from ..critical_bandwidth_fm import fisher_marron_critical_bandwidth, is_unimodal_kde as is_unimodal_kde_fm
from ..shoulder_distributions import bump_distribution
from ..util.bootstrap_MPI import probability_above
from ..util import print_rank0, print_all_ranks


class XSampleBW(XSample):

    def __init__(self, N, sampfun, comm=MPI.COMM_WORLD):
        super(XSampleBW, self).__init__(N, sampfun, comm)
        self.I = (-1.5, 1.5)  # avoiding spurious bumps in the tails
        self.h_crit = critical_bandwidth(self.data, self.I)
        #print_all_ranks(self.comm, "self.h_crit = {}".format(self.h_crit))
        self.var = np.var(self.data)
        self.kde_h_crit = KernelDensity(kernel='gaussian', bandwidth=self.h_crit).fit(self.data.reshape(-1, 1))

    @property
    def statistic(self):
        return self.h_crit

    def resampled_statistic_below_scaled_statistic(self, lambda_scale):
        '''
            P( h_{crit}^* <= \lambda*h_{crit})
                = P(KDE(X^*, \lambda* h_{crit}) is unimodal)
        '''
        return self.is_unimodal_resample(lambda_scale)

    def is_unimodal_resample(self, lambda_val):
        data = self.kde_h_crit.sample(self.N).reshape(-1)/np.sqrt(1+self.h_crit**2/self.var)
        #print "np.var(data)/self.var = {}".format(np.var(data)/self.var)
        return is_unimodal_kde(self.h_crit*lambda_val, data, self.I)

    def probability_of_unimodal_above(self, lambda_val, gamma):
        return self.prob_resampled_statistic_below_bound_above_gamma(lambda_val, gamma)
        #return probability_above(lambda: self.is_unimodal_resample(lambda_val),
        #                         gamma, max_samp=5000, comm=self.comm, batch=20)


class XSampleBwTrunc(XSampleBW):

    def __init__(self, N, sampfun, range_, comm=MPI.COMM_WORLD, blur_func=None):
        super(XSampleBwTrunc, self).__init__(N, sampfun, comm)
        #self.data = self.data[(self.data > -3) & (self.data < 3)]
        #print "nbr removed: {}".format(N-len(self.data))
        self.range_ = range_
        if blur_func is None:
            blur_func = lambda x: x
        self.blur_func = blur_func
        self.set_data_and_I(self.data, self.I)

    def set_data_and_I(self, data, I):
        self.data = np.round((data+3)*self.range_/6)
        self.I = [(i+3)*self.range_*1./6 for i in I]
        self.h_crit = critical_bandwidth(self.data, self.I)
        self.var = np.var(self.data)
        self.kde_h_crit = KernelDensity(kernel='gaussian', bandwidth=self.h_crit).fit(self.data.reshape(-1, 1))
        self.data = self.blur_func(self.data)      

    # def is_unimodal_resample(self, lambda_val):
    #     data = self.kde_h_crit.sample(self.N).reshape(-1)/np.sqrt(1+self.h_crit**2/self.var)
    #     #print "np.var(data)/self.var = {}".format(np.var(data)/self.var)
    #     return is_unimodal_kde(self.h_crit*lambda_val, self.blur_func(np.round(data)), self.I)


class XSampleShoulderBW(XSampleBW):
    '''
        Obsolete, use XSampleBW with sampfun='shoulder' instead.
    '''

    def __init__(self, N, comm=MPI.COMM_SELF):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.I = (-1.5, 1.5)  # CHECK: Is appropriate bound? OK.
        self.N = N
        if self.rank == 0:
            N1 = binom.rvs(N, 1.0/17)
            #print "N1 = {}".format(N1)
            N2 = N - N1
            m1 = -1.25
            s1 = 0.25
            data = np.hstack([s1*np.random.randn(N1)+m1, np.random.randn(N2)])
        else:
            data = None
        data = self.comm.bcast(data)
        self.data = data
        self.var = np.var(data)
        self.h_crit = critical_bandwidth(data, self.I)
        #print_all_ranks(self.comm, "self.h_crit = {}".format(self.h_crit))
        self.kde_h_crit = KernelDensity(kernel='gaussian', bandwidth=self.h_crit).fit(data.reshape(-1, 1))


def get_fm_sampling_class(mtol):

    a = bump_distribution(mtol, np.array([16./17, 1./17]), 0.25)

    class XSampleFMBW(XSampleBW):

        def __init__(self, N, comm=MPI.COMM_SELF):
            self.comm = comm
            self.rank = self.comm.Get_rank()
            self.I = (-1.5, a+1)  # CHECK: Is appropriate bound? OK.
            self.lamtol = 0
            self.mtol = mtol
            self.N = N
            if self.rank == 0:
                N1 = binom.rvs(N, 2.0/3)
                #print "N1 = {}".format(N1)
                N2 = N - N1
                data = np.hstack([np.random.randn(N1), np.random.randn(N2)+a])
            else:
                data = None
            data = self.comm.bcast(data)
            self.data = data
            self.var = np.var(data)
            self.h_crit = fisher_marron_critical_bandwidth(data, self.lamtol, self.mtol, self.I)
            #print_all_ranks(self.comm, "self.h_crit = {}".format(self.h_crit))
            self.kde_h_crit = KernelDensity(kernel='gaussian', bandwidth=self.h_crit).fit(data.reshape(-1, 1))

        def is_unimodal_resample(self, lambda_val):
            data = self.kde_h_crit.sample(self.N).reshape(-1)/np.sqrt(1+self.h_crit**2/self.var)
            #print "np.var(data)/self.var = {}".format(np.var(data)/self.var)
            return is_unimodal_kde_fm(self.h_crit*lambda_val, data, self.lamtol, self.mtol, self.I)

        def probability_of_unimodal_above(self, lambda_val, gamma):
            '''
                G_n(\lambda) = P(\hat h_{crit}^*/\hat h_{crit} <= \lambda)
                             = P(\hat h_{crit}^* <= \lambda*\hat h_{crit})
                             = P(KDE(X^*, \lambda*\hat h_{crit}) is unimodal)
            '''
            # print "bootstrapping 1000 samples at rank {}:".format(self.rank)
            # smaller_equal_crit_bandwidth = bootstrap(lambda: self.is_unimodal_resample(lambda_val), 1000, dtype=np.bool_)
            # pval = np.mean(~smaller_equal_crit_bandwidth)
            # print "result at rank {}: pval = {}".format(self.rank, pval)+"\n"+"-"*20
            return probability_above(lambda: self.is_unimodal_resample(lambda_val),
                                     gamma, max_samp=20000, comm=self.comm, batch=20)

    return XSampleFMBW


def get_sampling_class(null, **kwargs):
    if null == 'fm':
        return get_fm_sampling_class(**kwargs)
    sampling_dict = {'normal': XSampleBW, 'shoulder': XSampleShoulderBW}
    return sampling_dict[null]


def h_crit_scale_factor(alpha, null='normal', lower_lambda=0, upper_lambda=2.0,
                        comm=MPI.COMM_WORLD, save_file=None, **samp_class_args):

    rank = comm.Get_rank()
    sampling_class = get_sampling_class(null, **samp_class_args)

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
            lambda: sampling_class(N, comm=comm).probability_of_unimodal_above(
                lambda_val, 1-alpha), alpha, comm=MPI.COMM_SELF, batch=10, tol=0.005, print_per_batch=True)  # 0.005)

    def save_upper(lambda_bound):
        if null == 'fm':
            save_null = ('fm_{}'.format(samp_class_args['mtol']))
            save_lambda(lambda_bound, 'fm', save_null, alpha, upper=True, lambda_file=save_file)
        else:
            save_lambda(lambda_bound, 'bw', null, alpha, upper=True, lambda_file=save_file)

    def save_lower(lambda_bound):
        if null == 'fm':
            save_null = ('fm_{}'.format(samp_class_args['mtol']))
            save_lambda(lambda_bound, 'fm', save_null, alpha, upper=False, lambda_file=save_file)
        else:
            save_lambda(lambda_bound, 'bw', null, alpha, upper=False, lambda_file=save_file)

    lambda_tol = 1e-4

    N = 10000
    seed = np.random.randint(1000)
    seed = comm.bcast(seed)
    seed += rank
    #seed = 846
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


if __name__ == '__main__':
    if 0:
        print("h_crit_scale_factor(0.30, 0, 2.0) = {}".format(h_crit_scale_factor(0.30, 0, 2.0)))  # alpha=0.05 => lambda_alpha=1.12734985352

    if 1:
#        seed = np.random.randint(1000)
        seed = 851
        print("seed = {}".format(seed))
        np.random.seed(seed)
        xsamp = XSampleShoulderBW(10000)
        x = np.linspace(-2, 2, 200)
        fig, ax = plt.subplots()
        ax.plot(x, np.exp(xsamp.kde_h_crit.score_samples(x.reshape(-1, 1))))
        ax.axvline(-1.5)
        ax.axvline(1.5)
        kde_h = KernelDensity(kernel='gaussian', bandwidth=xsamp.h_crit*0.8).fit(xsamp.data.reshape(-1, 1))
        print("is_unimodal_kde(xsamp.h_crit*0.8, xsamp.data, (-1.5, 1.5)) = {}".format(is_unimodal_kde(xsamp.h_crit*0.8, xsamp.data, (-1.5, 1.5))))
        fig, ax = plt.subplots()
        ax.plot(x, np.exp(kde_h.score_samples(x.reshape(-1, 1))))
        ax.axvline(-1.5)
        ax.axvline(1.5)
        plt.show()

    if 0:
        XSampleBW(10000).is_unimodal_resample(1)


