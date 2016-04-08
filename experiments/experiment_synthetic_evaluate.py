import os
import numpy as np
import cPickle as pickle

if 1:
    resdir = '/Users/johnsson/Forskning/Experiments/modality/synthetic'
else:
    resdir = '/home/johnsson/Forskning/Experiments/modality/synthetic'

alpha = 0.05
median_summary = False
summary_file = os.path.join(resdir, 'summary{}_{}.csv'.format('_median'*median_summary, alpha))

with open(summary_file, 'w') as sumf:

    sumf.write('test, mtol, shoulder_ratio, N, ntest, {}\n'.format('frac_reject' if not median_summary else 'median'))

    for test in ['silverman', 'bandwidth_cal_normal', 'bandwidth_cal_shoulder']:

        resfile = os.path.join(resdir, test+'.pkl')

        with open(resfile, 'r') as resf:
            res = pickle.load(resf)

        for param in res:
            if test == 'silverman':
                ntest, mtol, shoulder_ratio, N = param
                if median_summary:
                    sum_stat = np.median(res[param])
                else:
                    sum_stat = np.mean(res[param] < alpha)  # rejection fraction
            else:
                ntest, mtol, shoulder_ratio, N, alpha_ = param
                if alpha != alpha_:
                    continue
                if median_summary:
                    sum_stat = np.median(res[param])
                else:
                    sum_stat = np.mean(res[param] < alpha)  # rejection fraction
            sumf.write('{}, {}, "{}/{}", {}, {}, {}\n'.format(test, mtol, shoulder_ratio[0], shoulder_ratio[1], N, ntest, sum_stat))

