import numpy as np
from scipy.stats import norm


class GaussianMixture1d(object):

    def __init__(self, means, covs, weights):
        self.mus = means
        self.sigmas = np.sqrt(covs)
        self.pis = weights
        self.kernel = lambda u: np.exp(-u**2/2.0)
        self._norm_factor = np.sqrt(2*np.pi)

    def evaluate_prop(self, x):  # returns values proportional to kde.
        xh = np.array(x).reshape(-1)
        res = np.zeros(len(xh))
        if len(xh) > len(self.mus):  # loop over data
            for mu, sigma, pi in zip(self.mus, self.sigmas, self.pis):
                res += pi/sigma*self.kernel((mu-xh)/sigma)
        else:  # loop over x
            for i, x_ in enumerate(xh):
                res[i] = np.sum(self.pis/self.sigmas*self.kernel((self.mus-x_)/self.sigmas))
        return res

    def evaluate(self, x):
        res = self.evaluate_prop(x)
        return res / self._norm_factor

    def score_samples(self, x):
        return np.log(self.evaluate(x))

    def distr(self, x):
        return np.sum(self.pis*norm.cdf((x - self.mus)/self.sigmas))

    def sample(self, n):
        ns = np.random.multinomial(n, self.pis)
        return np.hstack([np.random.randn(n_)*sigma+mu for n_, sigma, mu in zip(ns, self.sigmas, self.mus)])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    if 0:
        means = np.array([-1.25, 0])
        covs = np.array([0.25**2, 1])
        weights = np.array([1./17, 16./17])
    if 1:
        means = np.array([0, 0.735256195068])
        covs = np.array([1, 0.25**2])
        weights = np.array([16./17, 1./17])

    if 0:
        means, covs, weights = np.array([0]), np.array([2]), np.array([1]) 
    gm = GaussianMixture1d(means, covs, weights)
    dat = gm.sample(500000)
    q = -1
    print("np.mean(dat < q) = {}".format(np.mean(dat < q)))
    print("gm.distr(q) = {}".format(gm.distr(q)))
    plt.hist(dat, bins=100, normed=True)
    x = np.linspace(-3, 2.5, 100)
    plt.plot(x, gm.evaluate(x))
    plt.show()

