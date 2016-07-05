from mpi4py import MPI

from ..util.bootstrap_MPI import probability_above


class XSample(object):
    '''
        Class that samples a data set from a reference distribution,
        computes statistic.
    '''

    def __init__(self, N, sampfun, comm=MPI.COMM_WORLD):
        self.N = N
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.data = sampfun(N, self.comm)
        self.statistic = None

    def resampled_statistic_below_scaled_statistic(self, lambda_scale):
        pass

    def prob_resampled_statistic_below_bound_above_gamma(self, lambda_scale, gamma):
        '''
            Is the probability that a resampled statistic is below
            lambda_scale*(original statistic), significantly above gamma
            (returns True) or significantly below gamma (returns False)?
            Significance level of bound is 0.01. If after 5000 samples
            this cannot be determined, the result True/False is drawn
            by random with equal probabilities 0.5.

            Equivalently, with
            G_n(\lambda) = P(X^*/X <= \lambda),
            where X is statistic (\Delta or h_{crit}),
            is G_n(\lambda) significantly above or significantly below gamma?
        '''
        return probability_above(lambda: self.resampled_statistic_below_scaled_statistic(lambda_scale),
                                 gamma, max_samp=5000, comm=self.comm, batch=20,
                                 bound_significance=0.01)
