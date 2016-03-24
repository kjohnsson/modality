import os
import numpy as np
import cPickle as pickle

if 0:
    resdir = '/Users/johnsson/Forskning/Experiments/modality/synthetic'
else:
    resdir = '/home/johnsson/Forskning/Experiments/modality/synthetic'

alpha = 0.05
summary_file = os.path.join(resdir, 'summary_{}.csv'.format(alpha))

with open(summary_file, 'w') as sumf:

    sumf.write('test, mtol, shoulder_ratio, N, ntest, frac_reject\n')

    for test in ['silverman', 'bandwidth_cal_normal', 'bandwidth_cal_shoulder']:

        resfile = os.path.join(resdir, test+'.pkl')

        with open(resfile, 'r') as resf:
            res = pickle.load(resf)

        for param in res:
            if test == 'silverman':
                ntest, mtol, shoulder_ratio, N = param
                frac_reject = np.mean(res[param] < alpha)
            else:
                ntest, mtol, shoulder_ratio, N, alpha_ = param
                if alpha != alpha_:
                    continue
                frac_reject = np.mean(res[param])
            sumf.write('{}, {}, "{}/{}", {}, {}, {}\n'.format(test, mtol, shoulder_ratio[0], shoulder_ratio[1], N, ntest, frac_reject))

