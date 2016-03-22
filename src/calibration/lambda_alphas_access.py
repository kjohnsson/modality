import cPickle as pickle
import numpy as np
import os
from mpi4py import MPI

host = 'ke'

pkg_dirs = {'ke': '/Users/johnsson/Forskning/Code/modality',
            'au': '/lunarc/nobackup/users/johnsson/Simulations/modality',
            'ta': '/home/johnsson/Forskning/Code/modality'}

pkg_dir = pkg_dirs[host]
lambda_file = os.path.join(pkg_dir, 'src/calibration/lambda_alphas.pkl')


def load_lambdas(test, null, alpha):
    with open(lambda_file, 'r') as f:
        lambda_dict = pickle.load(f)

    return lambda_dict[test][null][alpha]


def load_lambda(test, null, alpha):
    lambdas = load_lambdas(test, null, alpha)
    return np.mean(lambdas)


def load_lambda_upper(test, null, alpha):
    lambdas = load_lambdas(test, null, alpha)
    return lambdas[1]


def load_lambda_lower(test, null, alpha):
    lambdas = load_lambdas(test, null, alpha)
    return lambdas[0]


def print_all_lambdas():
    with open(lambda_file, 'r') as f:
        lambda_dict = pickle.load(f)
    print "All computed lambda_alpha: {}".format(lambda_dict)


def save_lambda(lambda_val, test, null, alpha, upper):

    if MPI.COMM_WORLD.Get_rank() == 0:

        try:
            with open(lambda_file, 'r') as f:
                lambda_dict = pickle.load(f)
        except IOError:
            print "No file {}, starting from emtpy lambda_dict."
            lambda_dict = {}

        if not test in lambda_dict:
            lambda_dict[test] = {}

        if not null in lambda_dict[test]:
            lambda_dict[test][null] = {}

        if not alpha in lambda_dict[test][null]:
            lambda_dict[test][null][alpha] = np.nan*np.ones((2,))

        lambda_dict[test][null][alpha][np.int(upper)] = lambda_val

        with open(lambda_file, 'w') as f:
            pickle.dump(lambda_dict, f, -1)

        print "Saved {} as {} bound for test {} with null hypothesis {} at alpha = {}".format(
            lambda_val, 'upper' if upper else 'lower', test, null, alpha)

if __name__ == '__main__':
    
    if 0:
        with open(lambda_file, 'w') as f:
            pickle.dump({}, f, -1)

    if 0:
        for alpha in [0.05, 0.3]:
            ulam = np.load('../../upper_lambda_{}.npy'.format(alpha))
            save_lambda(ulam, 'bw', 'normal', alpha, True)
            llam = np.load('../../lower_lambda_{}.npy'.format(alpha))
            save_lambda(llam, 'bw', 'normal', alpha, False)

        alpha = 0.1
        llam, ulam = 1.25, 1.5
        save_lambda(ulam, 'bw', 'normal', alpha, True)
        save_lambda(llam, 'bw', 'normal', alpha, False)       

        print_all_lambdas()
