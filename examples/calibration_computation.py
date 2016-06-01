from mpi4py import MPI
import time
import traceback
import sys

host = 'au'

pkg_dirs = {'ke': '/Users/johnsson/Forskning/Code/modality',
            'au': '/lunarc/nobackup/users/johnsson/Simulations/modality',
            'ta': '/home/johnsson/Forskning/Code/modality'}

pkg_dir = pkg_dirs[host]

sys.path.append(pkg_dir)
from src.calibration.bandwidth import h_crit_scale_factor
from src.calibration.dip import dip_scale_factor_adaptive


def mpiexceptabort(type, value, tb):
    traceback.print_exception(type, value, tb)
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = mpiexceptabort


def main():
    t0 = time.time()

    #print "h_crit_scale_factor(0.10) = {}".format(h_crit_scale_factor(0.10))
    #print "h_crit_scale_factor(0.05, 'normal', lower_lambda=1, upper_lambda=1.0625) = {}".format(h_crit_scale_factor(0.3, 'normal', lower_lambda=1, upper_lambda=1.0625))
    #print "h_crit_scale_factor(0.05, 'shoulder') = {}".format(h_crit_scale_factor(0.05, 'shoulder'))
    #print "h_crit_scale_factor(0.05, 'fm', mtol=0.001, lower_lambda=1.125, upper_lambda=1.1875) = {}".format(
    #    h_crit_scale_factor(0.05, 'fm', mtol=0.001, lower_lambda=1.125, upper_lambda=1.1875))

    #print "dip_scale_factor(0.05, 'normal', lower_lambda=1.086328125, upper_lambda=1.0953125) = {}".format(
    #    dip_scale_factor(0.05, 'normal', lower_lambda=1.086328125, upper_lambda=1.0953125))
    print "dip_scale_factor_adaptive(0.05/6, 'shoulder', lower_lambda=1., upper_lambda=2.) = {}".format(
        dip_scale_factor_adaptive(0.05/6, 'shoulder', lower_lambda=1., upper_lambda=2.))

    t1 = time.time()

    print "Computation time: {}".format(t1-t0)

if __name__ == '__main__':
    main()
