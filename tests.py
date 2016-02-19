import numpy as np
import matplotlib.pyplot as plt

from src.diptest import cum_distr, dip_from_cdf, dip_pval_tabinterpol, transform_dip_to_other_nbr_pts
from src.modes import excess_mass_modes

tests = ['modes']

if 'modes' in tests:
    fig, axs = plt.subplots(3)
    for seed, ax in zip([None, 403, 796], axs):
        if seed is None:
            dat = np.hstack([np.arange(0, 1, .1), np.arange(2, 3, 0.1)])
        else:
            print "seed = {}".format(seed)
            np.random.seed(seed)
            dat = np.hstack([np.random.randn(100), np.random.randn(100)+4])
        w = np.ones(len(dat))*1./len(dat)
        xcum, ycum = cum_distr(dat, w)
        modes = excess_mass_modes(dat, w, 2)
        for mode in modes:
            # print "mode = {}".format(mode)
            # print "dat[mode[0]] = {}".format(dat[mode[0]])
            # print "dat[mode[1]] = {}".format(dat[mode[1]])
            ax.axvspan(dat[mode[0]], dat[mode[1]], color='grey')
        ax.hist(dat, weights=w)
    plt.show()

if 'diptestplot' in tests:
    for seed in [None, 403, 796]:
            if seed is None:
                dat = np.hstack([np.arange(0, 1, .1), np.arange(2, 3, 0.1)])
            else:
                print "seed = {}".format(seed)
                np.random.seed(seed)
                dat = np.hstack([np.random.randn(10), np.random.randn(10)+2])
            xcum, ycum = cum_distr(dat, np.ones(len(dat))*1./len(dat))
            dip = dip_from_cdf(xcum, ycum, verbose=True, plotting=True)
            print "dip = {}".format(dip)

if 'interpol' in tests:
    for (dip, N, M) in [(0.005, 20000, 50000), (0.01, 2000, 5000), (0.001, 70000, 10000), (0.0005, 1000000, 10000)]:
        print "dip_pval_tabinterpol(dip, N) = {}".format(dip_pval_tabinterpol(dip, N))
        print "dip_pval_tabinterpol(transform_dip_to_other_nbr_pts(dip, N, M), M) = {}".format(
            dip_pval_tabinterpol(transform_dip_to_other_nbr_pts(dip, N, M), M))
