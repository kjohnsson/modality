from mpi4py import MPI
import time
import traceback
import sys


def mpiexceptabort(type, value, tb):
    traceback.print_exception(type, value, tb)
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = mpiexceptabort

from lambda_alphas_calibration_bw import h_crit_scale_factor
from lambda_alphas_calibration_dip import dip_scale_factor

t0 = time.time()

#print "h_crit_scale_factor(0.10) = {}".format(h_crit_scale_factor(0.10))
#print "h_crit_scale_factor(0.05, 'normal', lower_lambda=1, upper_lambda=1.0625) = {}".format(h_crit_scale_factor(0.3, 'normal', lower_lambda=1, upper_lambda=1.0625))
#print "h_crit_scale_factor(0.05, 'shoulder') = {}".format(h_crit_scale_factor(0.05, 'shoulder'))
print "h_crit_scale_factor(0.05, 'fm', mtol=0.001, lower_lambda=1.125, upper_lambda=1.1875) = {}".format(
    h_crit_scale_factor(0.05, 'fm', mtol=0.001, lower_lambda=1.125, upper_lambda=1.1875))

#print "dip_scale_factor(0.05, 'normal', lower_lambda=1.086328125, upper_lambda=1.0953125) = {}".format(
#    dip_scale_factor(0.05, 'normal', lower_lambda=1.086328125, upper_lambda=1.0953125))
#print "dip_scale_factor(0.05, 'shoulder', lower_lambda=1.08125, upper_lambda=1.0875) = {}".format(
#    dip_scale_factor(0.05, 'shoulder', lower_lambda=1.08125, upper_lambda=1.0875))


t1 = time.time()

print "Computation time: {}".format(t1-t0)
