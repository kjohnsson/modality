from mpi4py import MPI
import numpy as np

from .reference_sampfun import normalsamp, shouldersamp, binom_confidence_interval
from .dip import XSampleDip
from .bandwidth import XSampleBW
from ..util.bootstrap_MPI import probability_in_interval
from .lambda_alphas_access import save_lambda
from ..util import print_rank0, print_all_ranks


def dip_scale_factor_adaptive(alpha, null='normal', lower_lambda=0, upper_lambda=2.0,
                              comm=MPI.COMM_WORLD):
    return calibration_scale_factor_adaptive(alpha, 'dip', null, lower_lambda,
                                             upper_lambda, comm)


def bw_scale_factor_adaptive(alpha, null='normal', lower_lambda=0, upper_lambda=2.0,
                             comm=MPI.COMM_WORLD):
    return calibration_scale_factor_adaptive(alpha, 'bw', null, lower_lambda,
                                             upper_lambda, comm)


def calibration_scale_factor_adaptive(alpha, type_, null='normal', lower_lambda=0, upper_lambda=2.0,
                                      comm=MPI.COMM_WORLD):
    '''
        Computing (and saving) the dip scale factor lambda_alpha for a
        test calibrated at level alpha.

            lower_lambda    -   lower bound for lambda_alpha in
                                bisection search.
            upper_lambda    -   upper bound for lambda_alpha in
                                bisection search.
    '''

    N_points = 10000
    rank = comm.Get_rank()
    nulldict = {'normal': normalsamp, 'shoulder': shouldersamp}
    typedict = {'dip': XSampleDip, 'bw': XSampleBW}
    sampfun = nulldict[null]
    xsample = typedict[type_]

    alpha_lower, alpha_upper = binom_confidence_interval(alpha, 5000, 0.1)
    print "(alpha_lower, alpha_upper) = {}".format((alpha_lower, alpha_upper))

    def rejection_rate_in_interval(lambda_, significance_first, significance_second):
        '''
            P(G_n(lambda) > 1-alpha) => reject null hypothesis
            G_n(lambda) = probability that resampled statistic is
            than lambda*(original statistic)
        '''
        print_rank0(comm, "Testing lambda_alpha = {}".format(lambda_))
        res = probability_in_interval(
            lambda: xsample(N_points, sampfun, comm=comm).prob_resampled_statistic_below_bound_above_gamma(
                lambda_, 1-alpha),
            alpha_lower, alpha_upper, significance_first=significance_first,
            significance_second=significance_second,
            comm=MPI.COMM_SELF, batch=20, print_per_batch=True)
        print_rank0(comm, "Rejection rate given lambda_val = {} is {}.".format(lambda_, res))
        return res

    def save_upper(lambda_bound):
        save_lambda(lambda_bound, type_+'_ad', null, alpha, upper=True)

    def save_lower(lambda_bound):
        save_lambda(lambda_bound, type_+'_ad', null, alpha, upper=False)

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
