import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import binom
import matplotlib.pyplot as plt

from .lambda_alphas_access import save_lambda
from ..bandwidth_test import is_unimodal_kde, critical_bandwidth
from ..bandwidth_fm_test import find_reference_distr, fisher_marron_critical_bandwidth, is_unimodal_kde as is_unimodal_kde_fm
from ..util.bootstrap_MPI import probability_above


class XSampleBW(object):

    def __init__(self, N):
        self.I = (-1.5, 1.5)  # avoiding spurious bumps in the tails
        self.N = N
        data = np.random.randn(N)
        self.var = np.var(data)
        self.h_crit = critical_bandwidth(data, self.I)
        self.kde_h_crit = KernelDensity(kernel='gaussian', bandwidth=self.h_crit).fit(data.reshape(-1, 1))

    def is_unimodal_resample(self, lambda_val):
        data = self.kde_h_crit.sample(self.N).reshape(-1)/np.sqrt(1+self.h_crit**2/self.var)
        #print "np.var(data)/self.var = {}".format(np.var(data)/self.var)
        return is_unimodal_kde(self.h_crit*lambda_val, data, self.I)

    def probability_of_unimodal_above(self, lambda_val, gamma):
        '''
            G_n(\lambda) = P(\hat h_{crit}^*/\hat h_{crit} <= \lambda)
                         = P(\hat h_{crit}^* <= \lambda*\hat h_{crit})
                         = P(KDE(X^*, \lambda*\hat h_{crit}) is unimodal)
        '''
        return probability_above(lambda: self.is_unimodal_resample(lambda_val), gamma, max_samp=5000, mpi=False)


class XSampleShoulderBW(XSampleBW):

    def __init__(self, N):
        self.I = (-1.5, 1.5)  # CHECK: Is appropriate bound? OK.
        self.N = N
        N1 = binom.rvs(N, 1.0/17)
        #print "N1 = {}".format(N1)
        N2 = N - N1
        m1 = -1.25
        s1 = 0.25
        data = np.hstack([s1*np.random.randn(N1)+m1, np.random.randn(N2)])
        self.data = data
        self.var = np.var(data)
        self.h_crit = critical_bandwidth(data, self.I)
        self.kde_h_crit = KernelDensity(kernel='gaussian', bandwidth=self.h_crit).fit(data.reshape(-1, 1))


def get_fm_sampling_class(mtol):

    a = find_reference_distr(mtol)

    class XSampleFMBW(XSampleBW):

        def __init__(self, N):
            self.I = (-1.5, a+1.5)  # CHECK: Is appropriate bound? OK.
            self.lamtol = 0
            self.mtol = mtol
            self.N = N
            N1 = binom.rvs(N, 2.0/3)
            #print "N1 = {}".format(N1)
            N2 = N - N1
            data = np.hstack([np.random.randn(N1), np.random.randn(N2)+a])
            self.data = data
            self.var = np.var(data)
            self.h_crit = fisher_marron_critical_bandwidth(data, self.lamtol, self.mtol, self.I)
            self.kde_h_crit = KernelDensity(kernel='gaussian', bandwidth=self.h_crit).fit(data.reshape(-1, 1))

        def is_unimodal_resample(self, lambda_val):
            data = self.kde_h_crit.sample(self.N).reshape(-1)/np.sqrt(1+self.h_crit**2/self.var)
            #print "np.var(data)/self.var = {}".format(np.var(data)/self.var)
            return is_unimodal_kde_fm(self.h_crit*lambda_val, data, self.lamtol, self.mtol, self.I)

    return XSampleFMBW


def get_sampling_class(null, **kwargs):
    if null == 'fm':
        return get_fm_sampling_class(**kwargs)
    sampling_dict = {'normal': XSampleBW, 'shoulder': XSampleShoulderBW}
    return sampling_dict[null]


def print_bound_search(fun):

    def printfun(lambda_val):
        print "Testing if {} is upper bound for lambda_alpha".format(lambda_val)
        res = fun(lambda_val)
        print "{} is".format(lambda_val)+" not"*(not res)+" upper bound for lambda_alpha."
        return res

    return printfun


def h_crit_scale_factor(alpha, null='normal', lower_lambda=0, upper_lambda=2.0, **samp_class_args):

    sampling_class = get_sampling_class(null, **samp_class_args)

    @print_bound_search
    def is_upper_bound_on_lambda(lambda_val):
        '''
            P(P(G_n(lambda)) > 1 - alpha) > alpha
                => lambda is upper bound on lambda_alpha
        '''
        return probability_above(lambda: sampling_class(N).probability_of_unimodal_above(lambda_val, 1-alpha), alpha, mpi=True, batch=20, tol=0.005)

    def save_upper(lambda_bound):
        save_lambda(lambda_bound, 'bw', null, alpha, upper=True)

    def save_lower(lambda_bound):
        save_lambda(lambda_bound, 'bw', null, alpha, upper=False)

    lambda_tol = 1e-4

    N = 10000
    #seed = np.random.randint(1000)
    seed = 846
    print "seed = {}".format(seed)
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
        print "h_crit_scale_factor(0.30, 0, 2.0) = {}".format(h_crit_scale_factor(0.30, 0, 2.0))  # alpha=0.05 => lambda_alpha=1.12734985352

    if 1:
#        seed = np.random.randint(1000)
        seed = 851
        print "seed = {}".format(seed)
        np.random.seed(seed)
        xsamp = XSampleShoulderBW(10000)
        x = np.linspace(-2, 2, 200)
        fig, ax = plt.subplots()
        ax.plot(x, np.exp(xsamp.kde_h_crit.score_samples(x.reshape(-1, 1))))
        ax.axvline(-1.5)
        ax.axvline(1.5)
        kde_h = KernelDensity(kernel='gaussian', bandwidth=xsamp.h_crit*0.8).fit(xsamp.data.reshape(-1, 1))
        print "is_unimodal_kde(xsamp.h_crit*0.8, xsamp.data, (-1.5, 1.5)) = {}".format(is_unimodal_kde(xsamp.h_crit*0.8, xsamp.data, (-1.5, 1.5)))
        fig, ax = plt.subplots()
        ax.plot(x, np.exp(kde_h.score_samples(x.reshape(-1, 1))))
        ax.axvline(-1.5)
        ax.axvline(1.5)
        plt.show()

    if 0:
        XSampleBW(10000).is_unimodal_resample(1)


