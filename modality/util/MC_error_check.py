from __future__ import unicode_literals
from __future__ import print_function
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
        if not np.isnan(relative_error) and relative_error > 1e-4:
            print("MC sampling relative error: {:.3f}, absolute_error: {:.3f}".format(relative_error, absolute_error))
        return res_mean

    return MCfun_check
