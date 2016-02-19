import numpy as np


def MC_error_check(MCfun):

    def MCfun_check(*args, **kwargs):
        N = 3
        res = np.zeros(N)
        for i in range(N):
            res[i] = MCfun(*args, **kwargs)
        res_mean = np.mean(res)
        absolute_error = np.max(np.abs(res-res_mean))
        relative_error = absolute_error/np.abs(res_mean)
        print "MC sampling relative error: {}".format(relative_error)
        print "MC sampling absolute error = {}".format(absolute_error)
        return res_mean

    return MCfun_check
