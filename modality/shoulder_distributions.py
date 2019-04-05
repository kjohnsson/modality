from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from scipy.optimize import leastsq
from scipy.stats import norm


def shoulder_density(x, w, m, s):
    return w[0]*np.exp(-x**2/2) + w[1]/s*np.exp(-((x-m)**2/(2*s**2)))


def shoulder_derivative(x, w, m, s):
    return -(w[0]*x*np.exp(-x**2/2)+w[1]/s**3*(x-m)*np.exp(-((x-m)**2/(2*s**2))))


def shoulder_derivative2(x, w, m, s):
    return w[0]*(x**2-1)*np.exp(-x**2/2)+w[1]/s**3*((x-m)**2/s**2-1)*np.exp(-((x-m)**2/(2*s**2)))


def shoulder_cdf(x, w, m, s):
    return w[0]*norm.cdf(x) + w[1]*norm.cdf(x, m, s)

'''
def shoulder_distribution(w, sd):
    a0 = 0.1
    maxiter = 30
    i = 0
    while i < maxiter:
        x0 = a0 if a0 > 1 else 0
        res = leastsq(lambda x, a: [shoulder_derivative(x, w, a, sd), shoulder_derivative2(x, w, a, sd)], x0=(x0, a0))
        x_a = res[0]
        x_shoulder = x_a[0]
        a_shoulder = x_a[1]
        if np.abs(shoulder_derivative(x_shoulder, w, a_shoulder, sd)) < 1e-4 and \
                np.abs(shoulder_derivative2(x_shoulder, w, a_shoulder, sd)) < 1e-4 and\
                np.abs(x_shoulder) < 10:
            return a_shoulder, x_shoulder
        a0 += 0.1
        i += 1
'''

def bump_size(w, m, sd, plot=False):
    x0s = np.linspace(0, m, 10)
    has_valley = False
    for x0 in x0s:
        x_valley = leastsq(lambda x: shoulder_derivative(x, w, m, sd), x0=x0)[0]
        if np.abs(shoulder_derivative(x_valley, w, m, sd)) < 1e-12 and \
                shoulder_derivative2(x_valley, w, m, sd) > 1e-12:
            has_valley = True
            break
    if not has_valley:
        return 0
    dens_valley = shoulder_density(x_valley, w, m, sd)
    # print "(x_valley, dens_valley) = {}".format((x_valley, dens_valley))

    x0s = np.linspace(-2*m, x_valley-1e-3, 10)
    for x0 in x0s:
        x_break_left = leastsq(lambda x: shoulder_density(x, w, m, sd)-dens_valley, x0=x0)[0]
        if np.abs(shoulder_density(x_break_left, w, m, sd)-dens_valley) < 1e-12 and \
                x_break_left < x_valley - 1e-3:
            break
    x0s = np.linspace(x_valley+1e-3, m*2, 10)
    for x0 in x0s:
        x_break_right = leastsq(lambda x: shoulder_density(x, w, m, sd)-dens_valley, x0=x0)[0]
        if np.abs(shoulder_density(x_break_right, w, m, sd)-dens_valley) < 1e-12 and \
                x_break_right > x_valley + 1e-3:
            break

    if plot:
        x_plot = np.linspace(-2*m, 2*m, 200)
        fig, ax = plt.subplots()
        ax.plot(x_plot, shoulder_density(x_plot, w, m, sd))
        ax.axvline(x_valley, color='red')
        ax.axvline(x_break_left)
        ax.axvline(x_break_right)
        ax.axhline(dens_valley)
        N = 2000
        x_rand = 4*m*(np.random.rand(N)-0.5)
        y_rand = 2*np.random.rand(N)
        below = y_rand < shoulder_density(x_rand, w, m, sd)
        ax.scatter(x_rand[below], y_rand[below])
        print("np.sum(below) = {}".format(np.sum(below)))
        plt.show()

    xs = [x_break_left, x_valley, x_break_right]
    cdfs = [shoulder_cdf(x, w, m, sd) for x in xs]
    return float(min(cdfs[2]-cdfs[1]-dens_valley*(xs[2]-xs[1])/np.sqrt(2*np.pi),
                 cdfs[1]-cdfs[0]-dens_valley*(xs[1]-xs[0])/np.sqrt(2*np.pi)))


def bump_distribution(msize, w, sd):
    a0 = 0.1
    maxiter = 30
    i = 0
    while i < maxiter:
        a_bump = leastsq(lambda a: bump_size(w, a, sd)-msize, x0=a0)[0]
        if np.abs(bump_size(w, a_bump, sd) - msize) < 1e-7:
            return a_bump
        a0 += 0.1
        i += 1

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    if 0:
        w = [0.8, 0.2]
        m = 0.9
        sd = 0.25
        print("bump_size(w, m, sd) = {}".format(bump_size(w, m, sd)))

        a = bump_distribution(0.005, w, sd)
        print("bump_size(w, a, sd, True) = {}".format(bump_size(w, a, sd, True)))

    if 0:
        shoulder_ratio = (16, 1)
        w = np.array(shoulder_ratio, dtype=np.float_)
        w /= np.sum(w)
        for s in [0.1, 0.25, 0.5, 1]:
            a_shoulder, x_shoulder = shoulder_distribution(w, s)
            print("a_shoulder = {}".format(a_shoulder))
            print("x_shoulder = {}".format(x_shoulder))
            print("shoulder_derivative(x_shoulder, w, a, s) = {}".format(shoulder_derivative(x_shoulder, w, a_shoulder, s)))
            print("shoulder_derivative2(x_shoulder, w, a, s) = {}".format(shoulder_derivative2(x_shoulder, w, a_shoulder, s)))

            x = np.linspace(-4, 4, 200)
            fig, axs = plt.subplots(1, 3, sharex=True, figsize=(16, 4))
            axs[0].plot(x, shoulder_density(x, w, a_shoulder, s))
            axs[1].plot(x, shoulder_derivative(x, w, a_shoulder, s))
            axs[2].plot(x, shoulder_derivative2(x, w, a_shoulder, s))
            for ax in axs.ravel():
                ax.axvline(x_shoulder, color='red')
                ax.axhline(0, color='gray')
        plt.show()

    if 0:
        latexplot = 1

        if latexplot:
            import matplotlib
            matplotlib.rc('text', usetex=True)
            matplotlib.rcParams['text.latex.preamble'] = ["\\usepackage\{amsmath\}"]
            matplotlib.rcParams['ps.useafm'] = True
            matplotlib.rcParams['pdf.use14corefonts'] = True
            matplotlib.rcParams['text.usetex'] = True

        sds = [0.1, 0.25, 0.5, 1]
        ratios = [(2, 1), (4, 1), (6, 1), (16, 1)]
        fig, axs = plt.subplots(4, 4, sharex=True, figsize=(6, 6))
        #plt.locator_params(axis='x', nbins=4)
        #plt.locator_params(axis='y', nbins=3)
        for shoulder_ratio, ax_row in zip(ratios, axs):
            w = np.array(shoulder_ratio, dtype=np.float_)
            w /= np.sum(w)
            ax_row[0].set_ylabel('$w_1\colon \! w_2 = {}\colon \!{}$'.format(*shoulder_ratio))
            for s, ax in zip(sds, ax_row):
                a_shoulder, x_shoulder = shoulder_distribution(w, s)
                print("a_shoulder = {}".format(a_shoulder))
                print("x_shoulder = {}".format(x_shoulder))
                print("shoulder_derivative(x_shoulder, w, a, s) = {}".format(shoulder_derivative(x_shoulder, w, a_shoulder, s)))
                print("shoulder_derivative2(x_shoulder, w, a, s) = {}".format(shoulder_derivative2(x_shoulder, w, a_shoulder, s)))

                x = np.linspace(-4, 4, 200)
                ax.plot(x, shoulder_density(x, w, a_shoulder, s))

                # ax.axvline(x_shoulder, color='red')
                # ax.axhline(0, color='gray')
        for ax, sd in zip(axs[0], sds):
            ax.set_title("$\sigma = {}$".format(sd))

        for ax in axs.ravel():
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_ticks([])
            #ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        fig.savefig('/Users/johnsson/Forskning/Experiments/modality/synthetic/shoulder_distributions.pdf', fmt='pdf',
                    bbox_inches='tight')
        fig.savefig('/Users/johnsson/Dropbox/Modality/figs/shoulder_distributions.pdf', fmt='pdf',
                    bbox_inches='tight')

        #plt.show()

    if 1:
        latexplot = 1

        if latexplot:
            import matplotlib
            matplotlib.rc('text', usetex=True)
            matplotlib.rcParams['text.latex.preamble'] = ["\\usepackage\{amsmath\}"]
            matplotlib.rcParams['ps.useafm'] = True
            matplotlib.rcParams['pdf.use14corefonts'] = True
            matplotlib.rcParams['text.usetex'] = True

        sds = [0.25, 0.5, 1]
        ratio = (4, 1)
        mtols = [0.01, 0.001, 0.0001, 0.000001]
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(4.5, 6))
        w = np.array(ratio, dtype=np.float_)
        w /= np.sum(w)
        for mtol, ax_row in zip(mtols, axs):
            ax_row[0].set_ylabel("$b = {}$".format(mtol))
            for sd, ax in zip(sds, ax_row):
                a_bump = bump_distribution(mtol, w, sd)
                print("a_bump = {}".format(a_bump))
                print("bump_size(w, a_bump, sd) = {}".format(bump_size(w, a_bump, sd)))

                x = np.linspace(-4, 4, 200)
                ax.plot(x, shoulder_density(x, w, a_bump, sd))

                # ax.axvline(x_shoulder, color='red')
                # ax.axhline(0, color='gray')
        for ax, sd in zip(axs[0], sds):
            ax.set_title('$\sigma = {}$'.format(sd))

        for ax in axs.ravel():
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_ticks([])
            #ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        fig.savefig('/Users/johnsson/Forskning/Experiments/modality/synthetic/bump_distributions.pdf', fmt='pdf',
                    bbox_inches='tight')
        fig.savefig('/Users/johnsson/Dropbox/Modality/figs/bump_distributions.pdf', fmt='pdf',
                    bbox_inches='tight')
        #plt.show()
