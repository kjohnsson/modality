from mpi4py import MPI
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

from ..diptest import cum_distr, dip_and_closest_unimodal_from_cdf, dip_from_cdf, sample_from_unimod
from ..util.bootstrap_MPI import bootstrap, bootstrap_array, probability_above, probability_in_interval
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
    '''
        Class that samples a data set from a reference distribution,
        computes dip and closest unimodal distribution and from which
        data from the closest unimodal distribution can be sampled.
    '''

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

    def prob_resampled_dip_below_bound_above_gamma(self, lambda_val, gamma):
        '''
            Is the probability that a resampled dip is below
            lambda_val*(original dip), significantly above gamma
            (returns True) or significantly below gamma (returns False)?
            Significance level of bound is 0.01. If after 5000 samples
            this cannot be determined, the result True/False is drawn
            by random with equal probabilities 0.5.

            Equivalently, with
            G_n(\lambda) = P(\hat Delta_{crit}^*/\hat Delta_{crit} <= \lambda)
            is G_n(\lambda) significantly above or significantly below gamma?
        '''
        return probability_above(lambda: self.dip_resampled() < lambda_val*self.dip,
                                 gamma, max_samp=5000, comm=self.comm, batch=20,
                                 bound_significance=0.01)

    def plot_unimodal(self):
        plt.plot(*self.unimod)


def binom_confidence_interval(alpha, N_discr, p_discr):
    '''
        Two-sided confidence interval of size 1-p_discr for binomial
        probability parameter given N_discr.

        Equivalently, using a two-sided test
        with significance level p_discr for alpha \\neq beta, the null
        hypothesis will not be rejected if beta is in the interval
        (lower, upper) and N_discr is the number of trials and
        beta*N_discr is the number of successfull tirals.

    '''
    lower = binom.ppf(p_discr/2, N_discr, alpha)*1./N_discr
    upper = binom.ppf(1-p_discr/2, N_discr, alpha)*1./N_discr
    return lower, upper


def dip_scale_factor_adaptive(alpha, null='normal', lower_lambda=0, upper_lambda=2.0,
                              comm=MPI.COMM_WORLD):
    '''
        Computing (and saving) the dip scale factor lambda_alpha for a
        test calibrated at level alpha.

            lower_lambda    -   lower bound for lambda_alpha in
                                bisection search.
            upper_lambda    -   upper bound for lambda_alpha in
                                bisection search.
    '''

    N_points = 100
    rank = comm.Get_rank()
    sampfun = normalsamp if null == 'normal' else shouldersamp

    alpha_lower, alpha_upper = binom_confidence_interval(alpha, 5000, 0.1)
    print "(alpha_lower, alpha_upper) = {}".format((alpha_lower, alpha_upper))

    def rejection_rate_in_interval(lambda_val, significance_first, significance_second):
        '''
            P(G_n(lambda) > 1-alpha) => reject null hypothesis
            G_n(lambda) = probability that resampled data has lower dip
            than lambda*(original dip)
        '''
        print_rank0(comm, "Testing lambda_alpha = {}".format(lambda_val))
        res = probability_in_interval(
            lambda: XSampleDip(N_points, sampfun, comm=comm).prob_resampled_dip_below_bound_above_gamma(
                lambda_val, 1-alpha),
            alpha_lower, alpha_upper, significance_first=significance_first,
            significance_second=significance_second,
            comm=MPI.COMM_SELF, batch=20, print_per_batch=True)
        print_rank0(comm, "Rejection rate given lambda_val = {} is {}.".format(lambda_val, res))
        return res

    def save_upper(lambda_bound):
        save_lambda(lambda_bound, 'dip_ex_ad', null, alpha, upper=True)

    def save_lower(lambda_bound):
        save_lambda(lambda_bound, 'dip_ex_ad', null, alpha, upper=False)

    seed = np.random.randint(1000)
    seed = comm.bcast(seed)
    seed += rank
    print_all_ranks(comm, "seed = {}".format(seed))
    np.random.seed(seed)

    lower_lambda = float(lower_lambda)
    upper_lambda = float(upper_lambda)

    while True:
        new_lambda = (upper_lambda+lower_lambda)/2
        rejection_rate_status = rejection_rate_in_interval(
            new_lambda, significance_first=0.01, significance_second=0.05)
        if rejection_rate_status == 'in interval':
            # alpha_lower < P(reject|lambda) < alpha_upper
            #  => lambda_alpha = lambda
            save_upper(new_lambda)
            save_lower(new_lambda)
            return new_lambda
        if rejection_rate_status == 'below upper bound':
            # P(reject|lambda) < alpha_upper => lambda_alpha >= lambda
            lower_lambda = new_lambda
            save_lower(lower_lambda)
            continue
        if rejection_rate_status == 'above lower bound':
            # P(reject|lambda) > alpha_lower => lambda_alpha <= lambda
            upper_lambda = new_lambda
            save_upper(upper_lambda)
            continue


def dip_scale_factor(alpha, null='normal', lower_lambda=0, upper_lambda=2.0,
                     comm=MPI.COMM_WORLD):

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
