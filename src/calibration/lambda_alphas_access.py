import cPickle as pickle
import numpy as np
import os
from mpi4py import MPI
import sys
import traceback


def mpiexceptabort(type, value, tb):
    traceback.print_exception(type, value, tb)
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = mpiexceptabort

lambda_dir = os.path.dirname(__file__)
lambda_file = os.path.join(lambda_dir, 'lambda_alphas.pkl')
lambda_bw_csv_file = os.path.join(lambda_dir, 'lambda_alphas_bw.csv')
lambda_dip_csv_file = os.path.join(lambda_dir, 'lambda_alphas_dip.csv')


def lambda_dict_to_csv():
    with open(lambda_file, 'r') as f:
        lambda_dict = pickle.load(f)

    with open(lambda_bw_csv_file, 'w') as f:
        f.write('Test, Null hypothesis, alpha, Lower bound, Upper bound\n')

        for test in lambda_dict:
            for null in lambda_dict[test]:
                for alpha in lambda_dict[test][null]:
                    lambdas = lambda_dict[test][null][alpha]
                    if not test == 'dip':
                        f.write('{}, {}, {}, {}, {}\n'.format(test, null, alpha, lambdas[0], lambdas[1]))

    with open(lambda_dip_csv_file, 'w') as f:
        f.write('Test, Null hypothesis, alpha, Lambda\n')

        for test in lambda_dict:
            for null in lambda_dict[test]:
                for alpha in lambda_dict[test][null]:
                    lambda_ = lambda_dict[test][null][alpha]
                    if test == 'dip':
                        f.write('{}, {}, {}, {}\n'.format(test, null, alpha, lambda_))


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


def save_lambda(lambda_val, test, null, alpha, upper=None):

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

        if test == 'bw' or test == 'bw_ad' or test == 'dip_ex' or test == 'fm' or test == 'dip_ex_ad' or 'dip_ad':
            if not alpha in lambda_dict[test][null]:
                lambda_dict[test][null][alpha] = np.nan*np.ones((2,))

            lambda_dict[test][null][alpha][np.int(upper)] = lambda_val
        elif test == 'dip':
            lambda_dict[test][null][alpha] = lambda_val
        else:
            raise ValueError('Unknown test: {}'.format(test))

        with open(lambda_file, 'w') as f:
            pickle.dump(lambda_dict, f, -1)

        lambda_dict_to_csv()

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

    if 1:
        lambda_dict_to_csv()
