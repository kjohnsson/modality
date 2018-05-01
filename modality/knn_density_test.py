'''
    Multidimensional test for multimodality, from
    Burman & Polonik 2009: Multivariate mode hunting: Data analytic
    tools with measures of significance. Journal of Multivariate
    Analysis 100 (2009).
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.neighbors import NearestNeighbors
from scipy.stats import f, norm, chi2
from scipy.special import gamma as gammafun


def pointwise_test(data, significance=0.05, standardize=False, plot=False):
    if standardize:
        data = standardize_mvn(data)
    n, p = data.shape
    k1, k2 = get_nbh_sizes(n, p)

    ## Step I: finding candidate modes
    nn = NearestNeighbors(k1, metric='euclidean').fit(data)
    possible_candidates = np.ones(n, dtype=np.bool)
    candidates = []
    while np.sum(possible_candidates) > 0:
        distances, indices = nn.kneighbors(data[possible_candidates])
        ind_new = np.argmin(distances[:, -1])
        new_candidate = np.arange(n)[possible_candidates][ind_new]
        candidates.append(new_candidate)
        possible_candidates[indices[ind_new, :k2]] = False

    ## Step II: Thin out candidates
    non_modes = []
    for i in candidates:
        mu = data[i, :]
        _, ind = nn.kneighbors(mu)
        ind = ind.ravel()
        X = data[ind[:k2], :]
        if hotelling_pval(X, mu) < 0.01:
            non_modes.append(i)
    modes = [i for i in candidates if not i in non_modes]

    ## Step III: SB-plot
    K = len(modes)
    in_other_modal_region = []
    for i in range(K):
        if i in in_other_modal_region:
            continue
        for j in range(i+1, K):
            if j in in_other_modal_region:
                continue
            x = data[modes[i], :]
            y = data[modes[j], :]
            alpha = np.linspace(0, 1, 200).reshape(-1, 1)
            x_alpha = alpha*x + (1-alpha)*y
            dist_k1nn, _ = nn.kneighbors(x_alpha)
            d_k1nn = dist_k1nn[:, -1]
            SB_alpha = p*(np.log(d_k1nn) - np.log(max(d_k1nn[0], d_k1nn[-1])))
            if (SB_alpha >= np.sqrt(2./k1)*norm.ppf(1-significance)).any() and plot:
                plt.plot(alpha, SB_alpha*np.sqrt(k1*1./2))
            else:
                in_other_modal_region.append(j)
    modal_regions = [mode for j, mode in enumerate(modes) if not j in in_other_modal_region]
    if len(modal_regions) > 1:
        return True
    return False


def standardize_mvn(data):
    '''
        Make data have mean 0 and covariance matrix I.
    '''
    X = (data - np.mean(data, axis=0)).T
    p, n = X.shape
    Sigma = np.cov(X)
    W = np.linalg.cholesky(np.linalg.inv(Sigma))
    Y = np.dot(W.T, X)
    return Y.T


def get_nbh_sizes(n, p):
    eta = chi2.ppf(0.95, p)
    c1 = (4*np.pi)**(-p*1./2)*chi2.cdf(2*eta, p)
    c21 = 1./(np.pi*(p+2))**2*(p*1./2*gammafun(p*1./2))**(4./p)
    N = 1000000
    chi2_rnd = chi2.rvs(p, size=N)
    c22 = (2*np.pi)**(-1./2+2./p)*np.mean((chi2_rnd <= eta)*(chi2_rnd-p)**2*np.exp(-(1./2-2./p)*chi2_rnd))
    c2 = c21*c22
    k1 = int(np.ceil(n**(4./(p+4))*(p*c1/c2)**(p*1./(p+4))))
    k2 = int(np.ceil(p*1./5*k1*np.sqrt(np.log(np.log(k1)))))
    return k1, k2


def hotelling_pval(X, mu):
    xbar = np.mean(X, axis=0)
    W = np.cov(X.T)
    n, p = X.shape
    t2 = n*np.dot(xbar-mu, np.linalg.solve(W, (xbar-mu).T))
    fstat = (n-p)*t2/p/(n-1)
    return 1-f.cdf(fstat, p, n-p)


if __name__ == '__main__':
    banknotes = pandas.read_csv('banknotes.csv')
    print banknotes.shape
    banknotes.head()
    plt.scatter(banknotes['Bottom'], banknotes['Diagonal'])
    data = np.hstack([banknotes['Bottom'].reshape(-1, 1), banknotes['Diagonal'].reshape(-1, 1)])
    print pointwise_test(data)

