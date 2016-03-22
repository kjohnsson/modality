import os
import cPickle as pickle

for test in ['silverman', 'bandwidth_cal_normal', 'bandwidth_cal_shoulder']:

    resdir = '/Users/johnsson/Forskning/Experiments/modality/synthetic'
    resfile = os.path.join(resdir, test+'.pkl')

    with open(resfile, 'r') as f:
        res = pickle.load(f)

    print "{} res = {}".format(test, res)
