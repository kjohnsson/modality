import numpy as np
from scipy.stats import binom


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