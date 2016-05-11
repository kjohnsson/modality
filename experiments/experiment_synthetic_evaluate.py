import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

if 1:
    resdir = '/Users/johnsson/Forskning/Experiments/modality/synthetic'
else:
    resdir = '/home/johnsson/Forskning/Experiments/modality/synthetic'

alpha = 0.05
median_summary = False
summary_file = os.path.join(resdir, 'summary_ref_shoulder{}_{}.csv'.format('_median'*median_summary, alpha))

with open(summary_file, 'w') as sumf:

    sumf.write('test, mtol, shoulder_ratio, shoulder_variance, N, I, ntest, {}\n'.format('frac_reject' if not median_summary else 'median'))

    for test in ['silverman', 'bandwidth_cal_normal', 'bandwidth_cal_shoulder',
                 'dip_nocal', 'dip_cal_normal', 'dip_cal_shoulder']:  # , 'bandwidth_cal_shoulder_ref', 'bandwidth_cal_shoulder_ref2']:

        resfile = os.path.join(resdir, test+'.pkl')

        with open(resfile, 'r') as resf:
            res = pickle.load(resf)

        for param in res:
            if test == 'silverman':
                if len(param) == 4:
                    ntest, mtol, shoulder_ratio, N = param
                    shoulder_variance = 1
                    I = (0, 0)
                elif len(param) == 5:
                    ntest, mtol, shoulder_ratio, shoulder_variance, N = param
                    I = (0, 0)
                elif len(param) == 6:
                    ntest, mtol, shoulder_ratio, shoulder_variance, N, I = param
                if median_summary:
                    sum_stat = np.median(res[param])
                else:
                    sum_stat = np.mean(res[param] < alpha)  # rejection fraction
            elif 'dip' in test:
                if len(param) == 5:
                    ntest, mtol, shoulder_ratio, shoulder_variance, N = param
                elif len(param) == 6:
                    ntest, mtol, shoulder_ratio, shoulder_variance, N, alpha_ = param
                I = (0, 0)
                if median_summary:
                    sum_stat = np.median(res[param])
                else:
                    sum_stat = np.mean(res[param] < alpha)  # rejection fraction
            else:
                if len(param) == 5:
                    ntest, mtol, shoulder_ratio, N, alpha_ = param
                    shoulder_variance = 1
                    I = (0, 0)
                elif len(param) == 6:
                    ntest, mtol, shoulder_ratio, shoulder_variance, N, alpha_ = param
                    I = (0, 0)
                elif len(param) == 7:
                    ntest, mtol, shoulder_ratio, shoulder_variance, N, I, alpha_ = param
                if alpha != alpha_:
                    continue
                if median_summary:
                    sum_stat = np.median(res[param])
                else:
                    #if shoulder_variance == 0.0625 and test == 'bandwidth_cal_shoulder_ref' and shoulder_ratio[0] == 16:
                    #    print "res[{}] = {}".format(param, res[param])
                        #plt.figure()
                        #plt.hist(res[param], bins=50)
                    sum_stat = np.mean(res[param] < alpha)  # rejection fraction
            if ntest == 500 and mtol == 0 and shoulder_variance == 0.25**2 and shoulder_ratio == (16, 1) and\
                    (I[0] in (-1.5, 0)) and (I[1] in (1.5, 0)):
                sumf.write('{}, {}, "{}/{}", {}, {}, "{}", {}, {}\n'.format(test, mtol, shoulder_ratio[0], shoulder_ratio[1], shoulder_variance, N, I, ntest, sum_stat))
plt.show()
