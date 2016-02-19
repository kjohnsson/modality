import cPickle as pickle
import numpy as np
from mpi4py import MPI

lambda_file = 'src/calibration/lambda_alphas.pkl'


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

        with open(lambda_file, 'r') as f:
            lambda_dict = pickle.load(f)

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
