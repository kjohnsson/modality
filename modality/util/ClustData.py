from __future__ import unicode_literals

from collections import Counter

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from . import MC_error_check


class ClustData(object):
    '''
        Object for handling clusters in data.
        Can find closest clusters in data set using
        Bhattacharyya distance.
    '''
    def __init__(self, data, labels, excludelab=None):
        if excludelab is None:
            excludelab = []
        self.label_names = [lab for lab in np.unique(labels) if not lab in excludelab]
        self.labels = labels
        self.K = len(self.label_names)
        self.data = data
        self.n, self.d = data.shape
        self._clusters = {}
        self.bhattacharyya_measure = bhattacharyya_coefficient

    def __iter__(self):
        return iter(self.label_names)

    def add_consensus_labelings(self, labels):
        consensus_labels = self.labels*0
        for k in range(self.K):
            cc = ConsensusCluster(self.labels, k)
            for lab in labels:
                cc.add_labeling(lab)
            consensus_labels[cc.in_cluster] = k
        self.labels = consensus_labels

    def cluster(self, k):
        try:
            return self._clusters[k]
        except KeyError:
            self._clusters[k] = self.data[self.labels == k, :]
        return self._clusters[k]

    def in_cluster(self, ks):
        incl = np.zeros((self.n, len(ks)), dtype='bool')
        for i, k in enumerate(ks):
            incl[self.labels == k] = 1
        return np.any(incl, axis=1)

    def get_closest(self, n):
        ind_1d = np.argpartition(-self.bhattacharyya_coefficient_toother, n,
                                 axis=None)[:n]
        ind_1d = ind_1d[self.bhattacharyya_coefficient_toother.ravel()[ind_1d] > 0]
        ind = np.unravel_index(ind_1d, (self.K, self.K))
        return zip([self.label_names[i] for i in ind[0]],
                   [self.label_names[i] for i in ind[1]])[:n]

    def most_discriminating_dim(self, k1, k2):
        bhd_1d = np.zeros(self.d)
        for dd in range(self.d):
            bhd_1d[dd] = self.bhattacharyya_measure(
                self.cluster(k1)[:, dd], self.cluster(k2)[:, dd])
        return np.argmin(bhd_1d)

    def split_in_other_labelings(self, k1, k2, labelings):
        '''
            Check if most common label in k1 (by other labelings) is
            different from most common label in k2.
        '''
        diffs = 0
        for label in labelings:
            most_common_k1 = Counter(label[self.labels == k1]).most_common()[0][0]
            most_common_k2 = Counter(label[self.labels == k2]).most_common()[0][0]
            if most_common_k1 != most_common_k2:
                diffs += 1
        return diffs

    def scatterplot_most_discriminating_dim(self, k1, k2, axs):
        dim = self.most_discriminating_dim(k1, k2)
        for d, ax in enumerate(axs):
            self.scatterplot([k1, k2], [d, dim], ax)

    def hist2d_most_discriminating_dim(self, k1, k2, axs, **figargs):
        dim = self.most_discriminating_dim(k1, k2)
        for d, ax in enumerate(axs):
            self.hist2d([k1, k2], [d, dim], ax, **figargs)

    def scatterplot(self, ks, dim, ax):
        cmap = plt.get_cmap('gist_rainbow')
        K = len(ks)
        colors = [cmap((0.2+k)*1./(K-1)) for k in range(K)]
        for k, color in zip(ks, colors):
            ax.scatter(self.cluster(k)[:, dim[0]], self.cluster(k)[:, dim[1]],
                       color=color, marker='+')

    def hist2d(self, ks, dim, ax, **figargs):
        data = np.vstack([self.cluster(k) for k in ks])
        ax.hist2d(data[:, dim[0]], data[:, dim[1]], **figargs)

    def boxplot_closest(self, n):
        closest = self.get_closest(n)
        n = len(closest)
        fig, axs = plt.subplots(n, squeeze=False, figsize=(4, (n-1)*1.3+1))
        for ax, ind in zip(axs.ravel(), closest):
            for k in ind:
                ax.boxplot(np.hsplit(self.cluster(k), self.d))
            ax.set_title('Cluster {} and {}'.format(*ind))

    def hist_closest(self, n):
        closest = self.get_closest(n)
        n = len(closest)
        fig, axs = plt.subplots(n, self.d, squeeze=False, figsize=(4+(self.d-1)*2, (n-1)*1.3+1))
        for ax_c, ind in zip(axs, closest):
            ranges = zip(np.minimum(np.min(self.cluster(ind[0]), axis=0), np.min(self.cluster(ind[1]), axis=0)),
                         np.maximum(np.max(self.cluster(ind[0]), axis=0), np.max(self.cluster(ind[1]), axis=0)))
            for dd, (ax, range_) in enumerate(zip(ax_c, ranges)):
                for color, k in zip(['blue', 'red'], ind):
                    ax.hist(self.cluster(k)[:, dd], bins=20, range=range_, color=color, alpha=0.6)
                #ax.set_ylim(0, 200)
                ax.set_title('Cluster {} and {}'.format(*ind))

    @property
    def bhattacharyya_coefficient_toother(self):
        try:
            return self._bhattacharyya_coefficient_toother
        except AttributeError:
            bdb = np.zeros((self.K, self.K))
            for i, k in enumerate(self):
                for j, kk in enumerate(self):
                    if j <= i:
                        continue
                    bdb[i, j] = self.bhattacharyya_measure(
                        self.cluster(k), self.cluster(kk))
            self._bhattacharyya_coefficient_toother = bdb
            return bdb

    @property
    def bhattacharyya_coefficient_toself(self):
        try:
            return self._bhattacharyya_coefficient_toself
        except AttributeError:
            bdw = np.zeros(self.K)
            for i, k in enumerate(self):
                bdw[i] = self.bhattacharyya_measure(
                    self.cluster(k), self.cluster(k))
            self._bhattacharyya_coefficient_toself = bdw
            return bdw

    @property
    def bhattacharyya_distances(self):
        bhattacharyya_coefficients = (
            self.bhattacharyya_coefficient_toother +
            self.bhattacharyya_coefficient_toother.T +
            np.diag(self.bhattacharyya_coefficient_toself))
        return -np.log(bhattacharyya_coefficients)

    def plot_bhattacharrya(self):
        plt.matshow(self.bhattacharyya_distances)


class ConsensusCluster(object):
    '''
        For finding a cluster that is common across a number of
        labelings.
    '''

    def __init__(self, labels, k):
        self.in_cluster = labels == k

    @property
    def size(self):
        return np.sum(self.in_cluster)

    def add_labeling(self, labels):
        k = Counter(labels[self.in_cluster]).most_common(1)[0][0]
        self.in_cluster *= labels == k
        return k

    def select_data(self, data):
        return data[self.in_cluster, :]

    def hist(self, data, bounds=(-np.inf, np.inf), fig=None):
        d = data.shape[1]
        data_cc = self.select_data(data)
        if fig is None:
            fig = plt.figure()
        for dd in range(d):
            ax = fig.add_subplot(1, d, dd+1)
            data_cc_d = data_cc[:, dd]
            ax.hist(data_cc_d[(data_cc_d > bounds[0])*(data_cc_d < bounds[1])], bins=100)

        for ax in fig.axes:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    def hist2d(self, data, fig=None):
        d = data.shape[1]
        data_cc = self.select_data(data)
        if fig is None:
            fig = plt.figure()
        for dd in range(d):
            for ddd in range(dd+1, d):
                ax = fig.add_subplot(d, d, dd*d+ddd+1)
                ax.hist2d(data_cc[:, dd], data_cc[:, ddd], bins=30,
                          norm=colors.LogNorm(), vmin=1)
                ax.set_xlim(np.min(data[:, dd]), np.max(data[:, dd]))
                ax.set_ylim(np.min(data[:, ddd]), np.max(data[:, ddd]))

    def scatter_data(self, data):
        d = data.shape[1]
        data_cc = self.select_data(data)
        fig = plt.figure()
        for dd in range(d):
            for ddd in range(dd+1, d):
                ax = fig.add_subplot(d, d, dd*d+ddd+1)
                ax.scatter(data_cc[:, dd], data_cc[:, ddd], marker='+')
                ax.set_xlim(np.min(data[:, dd]), np.max(data[:, dd]))
                ax.set_ylim(np.min(data[:, ddd]), np.max(data[:, ddd]))


def bhattacharyya_coefficient_discrete(data1, data2, bins=10):
    '''
        Computing Bhattacharyya coefficient using (multidimensional)
        histograms.
    '''
    hist_range = zip(np.minimum(np.min(data1, axis=0), np.min(data2, axis=0)),
                     np.maximum(np.max(data1, axis=0), np.max(data2, axis=0)))
    bins_total_volume = np.prod([ma-mi for mi, ma in hist_range])

    hist1, _ = np.histogramdd(data1, bins=bins, range=hist_range, normed=True)
    hist2, _ = np.histogramdd(data2, bins=bins, range=hist_range, normed=True)

    return np.mean(np.sqrt(hist1*hist2))*bins_total_volume


@MC_error_check
def bhattacharyya_coefficient(data1, data2, N=1000):
    '''
        Computing Bhattacharyya coefficient (using MC sampling)
        between kernel density estimates of data with bandwith
        selection by Scott's rule.
    '''

    try:
        d = data1.shape[1]
    except IndexError:
        d = 1
    if data1.shape[0] < d or data2.shape[0] < d:
        return 0

    try:
        kde1 = gaussian_kde(data1.T)
        kde2 = gaussian_kde(data2.T)
    except np.linalg.linalg.LinAlgError:
        return 0
    samp1 = kde1.resample(N/2)
    samp2 = kde2.resample(N/2)

    return (np.mean(np.sqrt(kde2(samp1)/kde1(samp1))) +
            np.mean(np.sqrt(kde1(samp2)/kde2(samp2))))/2