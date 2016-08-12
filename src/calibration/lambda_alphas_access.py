import cPickle as pickle
import numpy as np
import os
from mpi4py import MPI


lambda_dir = os.path.join(os.path.dirname(__file__), 'data')
lambda_file_precomputed = os.path.join(lambda_dir, 'lambda_alphas.pkl')


def load_lambdas(test, null, alpha, lambda_file=None):
    if lambda_file is None:
        lambda_file = lambda_file_precomputed
    with open(lambda_file, 'r') as f:
        lambda_dict = pickle.load(f)

    return lambda_dict[test][null][alpha]


def load_lambda(test, null, alpha, lambda_file=None):
    if lambda_file is None:
        lambda_file = lambda_file_precomputed
    lambdas = load_lambdas(test, null, alpha, lambda_file)
    return np.mean(lambdas)


def load_lambda_upper(test, null, alpha, lambda_file=None):
    if lambda_file is None:
        lambda_file = lambda_file_precomputed
    lambdas = load_lambdas(test, null, alpha, lambda_file)
    return lambdas[1]


def load_lambda_lower(test, null, alpha, lambda_file=None):
    if lambda_file is None:
        lambda_file = lambda_file_precomputed
    lambdas = load_lambdas(test, null, alpha, lambda_file)
    return lambdas[0]


def save_lambda(lambda_val, test, null, alpha, upper=None, lambda_file=None):
    if lambda_file is 'precomputed':
        lambda_file = lambda_file_precomputed

    if MPI.COMM_WORLD.Get_rank() == 0:

        try:
            with open(lambda_file, 'r') as f:
                lambda_dict = pickle.load(f)
        except (IOError, EOFError):
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

        print "Saved {} as {} bound for test {} with null hypothesis {} at alpha = {}".format(
            lambda_val, 'upper' if upper else 'lower', test, null, alpha)


def print_computed_calibration(lambda_file=None, include_dip_approx=False):
    if lambda_file is None:
        lambda_file = lambda_file_precomputed
    print "lambda_alpha in {}:".format(lambda_file)
    print lambda_dict_to_csv(lambda_file)

    if include_dip_approx:
        print "Approximate lambda_alpha in {}".format(lambda_file)
        print lambda_dict_dip_approx_to_csv(lambda_file)


def lambda_dict_to_csv(lambda_file=None):
    if lambda_file is None:
        lambda_file = lambda_file_precomputed
    with open(lambda_file, 'r') as f:
        lambda_dict = pickle.load(f)

    csv_str = 'Test, Null hypothesis, alpha, Lower bound, Upper bound\n'

    for test in lambda_dict:
        for null in lambda_dict[test]:
            for alpha in lambda_dict[test][null]:
                lambdas = lambda_dict[test][null][alpha]
                if not test == 'dip':
                  # The key 'dip' represents approximately computed constants
                    csv_str += '{}, {}, {}, {}, {}\n'.format(test, null, alpha, lambdas[0], lambdas[1])

    return csv_str


def lambda_dict_dip_approx_to_csv(lambda_file=None):
    if lambda_file is None:
        lambda_file = lambda_file_precomputed

    with open(lambda_file, 'r') as f:
        lambda_dict = pickle.load(f)

    csv_str = 'Test, Null hypothesis, alpha, Lambda\n'

    for test in lambda_dict:
        for null in lambda_dict[test]:
            for alpha in lambda_dict[test][null]:
                lambda_ = lambda_dict[test][null][alpha]
                if test == 'dip':
                    csv_str += '{}, {}, {}, {}\n'.format(test, null, alpha, lambda_)

    return csv_str

if __name__ == '__main__':

    if 1:
        print_computed_calibration()
