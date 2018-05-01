'''
Based on Bruce E. Hansen 2009: Lecture Notes on Nonparametrics,
http://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf.
'''
import numpy as np
import matplotlib.pyplot as plt


class KernelDensityDerivative(object):

    def __init__(self, data, deriv_order):

        if deriv_order == 0:
            self.kernel = lambda u: np.exp(-u**2/2)
        elif deriv_order == 2:
            self.kernel = lambda u: (u**2-1)*np.exp(-u**2/2)
        else:
            raise ValueError('Not implemented for derivative of order {}'.format(deriv_order))
        self.deriv_order = deriv_order
        self.h = silverman_bandwidth(data, deriv_order)
        self.datah = data/self.h

    def evaluate(self, x):
        xh = np.array(x).reshape(-1)/self.h
        res = np.zeros(len(xh))
        if len(xh) > len(self.datah):  # loop over data
            for data_ in self.datah:
                res += self.kernel(data_-xh)
        else:  # loop over x
            for i, x_ in enumerate(xh):
                res[i] = np.sum(self.kernel(self.datah-x_))
        return res*1./(np.sqrt(2*np.pi)*self.h**(1+self.deriv_order)*len(self.datah))

    def score_samples(self, x):
        return self.evaluate(x)

    def plot(self, ax=None):
        x = self.h*np.linspace(np.min(self.datah)-5, np.max(self.datah)+5, 200)
        y = self.evaluate(x)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y)


def silverman_bandwidth(data, deriv_order=0):
    sigmahat = np.std(data, ddof=1)
    return sigmahat*bandwidth_factor(data.shape[0], deriv_order)


def bandwidth_factor(nbr_data_pts, deriv_order=0):
    '''
        Scale factor for one-dimensional plug-in bandwidth selection.
    '''
    if deriv_order == 0:
        return (3.0*nbr_data_pts/4)**(-1.0/5)

    if deriv_order == 2:
        return (7.0*nbr_data_pts/4)**(-1.0/9)

    raise ValueError('Not implemented for derivative of order {}'.format(deriv_order))


if __name__ == '__main__':
    print "bandwidth_factor(1, 0) = {}".format(bandwidth_factor(1, 0))
    print "bandwidth_factor(1, 2) = {}".format(bandwidth_factor(1, 2))

    data = np.random.randn(100)
    fig, axs = plt.subplots(1, 2)
    KernelDensityDerivative(data, 0).plot(axs[0])
    KernelDensityDerivative(data, 2).plot(axs[1])
    KernelDensityDerivative(data, 0).plot()
    plt.show()
