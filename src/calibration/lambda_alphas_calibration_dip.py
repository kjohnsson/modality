import numpy as np
from collections import Counter
from scipy.stats import binom
import matplotlib.pyplot as plt

from ..diptest import cum_distr, dip_and_closest_unimodal_from_cdf, dip_from_cdf
from ..util.bootstrap_MPI import bootstrap_mpi, bootstrap, bootstrap_array


class XSampleDip(object):

    def __init__(self, N):
        self.N = N
        data = np.random.randn(N)
        data = data[np.abs(data) < 1.5]  # avoiding spurious bumps in the tails
        xF, yF = cum_distr(data)
        self.dip, self.unimod = dip_and_closest_unimodal_from_cdf(xF, yF)

    def dip_resampled(self):
        data = self.sample_from_unimod()
        xF, yF = cum_distr(data)
        return dip_from_cdf(xF, yF)

    def sample_from_unimod(self):
        xU, yU = self.unimod
        #print "zip(xU, yU) = {}".format(zip(xU, yU))
        dxU = np.diff(xU)
        xU = np.hstack([xU[0], xU[1:][dxU > 0]])
        yU = np.hstack([yU[0], yU[1:][dxU > 0]])
        dxU = dxU[dxU > 0]
        #print "zip(xU, yU) = {}".format(zip(xU, yU))
        kU = np.diff(yU)/np.diff(xU)
        cum_probU = np.cumsum(kU*np.diff(xU))
        t = np.random.rand(self.N)
        bins = np.searchsorted(cum_probU, t)
        bin_cnt = Counter(bins).most_common()
        data = np.zeros((self.N,))
        i = 0
        for bin, cnt in bin_cnt:
            data[i:i+cnt] = np.random.rand(cnt)*dxU[bin]+xU[bin]
            i += cnt
        return data

    def lowest_lambdas_rejecting(self, alphas, B=50):
        dips = bootstrap_mpi(self.dip_resampled, B)
        i_s = np.floor(alphas*B)
        dip_thrs = -np.sort(-dips)[i_s.astype(np.int)]
        lambdas = dip_thrs/self.dip
        #print "np.mean(dips/self.dip <= lambd) = {}".format(np.mean(dips/self.dip <= lambd))
        return lambdas

    def plot_unimodal(self):
        plt.plot(*self.unimod)


def dip_scale_factor(alphas, B=50, N=10000):
    i_s = np.round(B*alphas)-1
    lowest_lambdas_rejects = bootstrap_array(lambda: XSampleDip(N).lowest_lambdas_rejecting(alphas), B, len(alphas))
    lambdas = -np.sort(-lowest_lambdas_rejects, axis=0)[i_s.astype(np.int), np.arange(len(alphas))]
    #print "np.mean(lowest_lambdas_rejecting <= lambd) = {}".format(np.mean(lowest_lambdas_rejecting <= lambd))
    return lambdas


if __name__ == '__main__':
    #seed = np.random.randint(1000)
    import time
    seed = 411
    print "seed = {}".format(seed)
    np.random.seed(seed)
    if 0:
        xdip = XSampleDip(1000)
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
    lambda_alphas = dip_scale_factor(alphas)
    t1 = time.time()
    print "Time: {}".format(t1-t0)
    plt.plot(alphas, lambda_alphas)
    plt.show()
