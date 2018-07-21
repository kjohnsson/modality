from __future__ import unicode_literals
from mpi4py import MPI

from .adaptive_calibration import calibration_scale_factor_adaptive
from .dip import dip_scale_factor
from .bandwidth import h_crit_scale_factor


def compute_calibration(calibration_file, test, null, alpha, adaptive=True,
                        lower_lambda=0, upper_lambda=2.0, comm=MPI.COMM_WORLD):
    '''
        Compute calibration constant lambda_alpha and save to file
        'calibration_file'.

        Input:
            test            -   'dip' or 'bw'.
            null            -   'shoulder' or 'normal'. Reference
                                distribution.
            alpha           -   significance level.
            adaptive        -   should adaptive probabilistic bisection
                                search be used?
            lower_lambda    -   lower bound for lambda_alpha in
                                bisection search.
            upper_lambda    -   upper bound for lambda_alpha in
                                bisection search.
            comm            -   MPI communicator.
    '''

    if comm.Get_rank() == 0:
        try:
            with open(calibration_file, 'a') as f:
                pass  # check that it is possible to write to file
        except Exception as e:
            exc = e
        else:
            exc = None
    else:
        exc = None
    exc = comm.bcast(exc)
    if not exc is None:
        raise exc

    if adaptive:
        return calibration_scale_factor_adaptive(alpha, test, null, lower_lambda, upper_lambda,
                                                 comm, calibration_file)

    if test == 'dip':
        return dip_scale_factor(alpha, null, lower_lambda, upper_lambda,
                                comm, calibration_file)

    if test == 'bw':
        return h_crit_scale_factor(alpha, null, lower_lambda, upper_lambda,
                                   comm, calibration_file)