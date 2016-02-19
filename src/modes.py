import numpy as np

from .diptest import dip_and_closest_unimodal_from_cdf, cum_distr


def excess_mass_modes(data, w, n):
    x_F, y_F = cum_distr(data, w)
    dip, (x_uni, y_uni) = dip_and_closest_unimodal_from_cdf(x_F, y_F)
    print "dip = {}".format(dip)
    D_max = -np.inf
    for lambd in diff(y_uni)/diff(x_uni):
    #lambd = 0.15
        #print "lambd = {}".format(lambd)
        D, modes = excess_mass_modes_lambda(data, w, n, lambd)
        if D > D_max:
            D_max = D
            best_modes = modes
    print "D_max = {}".format(D_max)
    return best_modes


def excess_mass_modes_lambda(data, w, n, lambd):
    order = np.argsort(data)
    H_lambda = np.cumsum(w[order]) - lambd*data[order]
    intervals = LinkedIntervals((0, len(order)))
    while len(intervals) < 2*n+1:
        #print "[interval.data for interval in intervals] = {}".format([interval.data for interval in intervals])
        E_max_next = -np.inf
        for i, interval_item in enumerate(intervals):
            if np.mod(i, 2) == 0:
                E_new, interval_new = find_mode(H_lambda, *interval_item.data)
            else:
                E_new, interval_new = find_antimode(H_lambda, *interval_item.data)
            if E_new > E_max_next:
                E_max_next = E_new
                interval_best = interval_new
                i_best = i
        intervals.split(i_best, interval_best)
    #print "[interval.data for interval in intervals] = {}".format([interval.data for interval in intervals])
    return E_max_next, [(order[interval.data[0]], order[interval.data[1]-1]) for i, interval in enumerate(intervals) if np.mod(i, 2) == 1]


def find_mode(H_lambda, i_lower, i_upper):
    if i_lower == i_upper:
        return 0, (i_lower, i_upper)
    H_min = np.inf
    E_max = -np.inf
    for i in range(i_lower, i_upper):
        if H_lambda[i] < H_min:
            H_min = H_lambda[i]
            i_mode_lower_curr = i
        E_new = H_lambda[i]-H_min
        if E_new > E_max:
            E_max = E_new
            i_mode_upper = i+1
            i_mode_lower = i_mode_lower_curr
    return E_max, (i_mode_lower, i_mode_upper)


def find_antimode(H_lambda, i_lower, i_upper):
    E, interval = find_mode(-H_lambda, i_lower, i_upper)
    return -E, interval


class LinkedList(object):
    def __init__(self, data=None):
        self._len = 0
        if not data is None:
            self.append(data)

    def __len__(self):
        return self._len

    def __iter__(self):
        return LinkedListIterator(self)

    def append(self, data):
        if len(self) > 0:
            self.last.next = ListItem(data)
        else:
            self.first = ListItem(data)
            self.before_first = ListItem(None)
            self.before_first.next = self.first
            self.last = self.first
            self._len += 1


class ListItem(object):
    def __init__(self, data):
        self.data = data

    # @property
    # def next(self):
    #     try:
    #         return self._next
    #     except AttributeError:
    #         raise OutsideListError

    # @next.setter
    # def next(self, next):
    #     self._next = next


class LinkedListIterator(object):
    def __init__(self, linked_list):
        self.curr = linked_list.before_first

    def __iter__(self):
        return self

    def next(self):
        try:
            self.curr = self.curr.next
        except AttributeError:
            raise StopIteration
        return self.curr


class LinkedIntervals(LinkedList):

    def split(self, i, insert_interval):
        for j, interval in enumerate(self):
            if j == i:
                try:
                    next = interval.next
                except AttributeError:
                    no_next = True
                else:
                    no_next = False
                imin, imax = interval.data
                insertmin, insertmax = insert_interval
                interval.data = imin, insertmin
                interval.next = ListItem((insertmin, insertmax))
                interval.next.next = ListItem((insertmax, imax))
                if not no_next:
                    interval.next.next.next = next
                self._len += 2
                return


def diff(x):
    N = len(x)
    dmat = np.eye(N) + np.diag(np.ones(N-1), -1)
    return dmat[1:, :].dot(x)
