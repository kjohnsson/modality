from __future__ import unicode_literals
import unittest
import tempfile
import os
import numpy as np
from mpi4py import MPI

from modality.calibration import compute_calibration, print_computed_calibration
from modality import calibrated_diptest
from modality.calibration.lambda_alphas_access import save_lambda


class TestCalibration(unittest.TestCase):

    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        if self.rank == 0:
            f, self.calibration_file = tempfile.mkstemp()
            os.close(f)
        else:
            self.calibration_file = None
        self.calibration_file = self.comm.bcast(self.calibration_file)

    def test_calibration_badfile(self):
        alpha = 0.3
        null = 'shoulder'
        self.assertRaises(Exception, compute_calibration, 'unvalid/dir/calfile.pkl', 'dip', null, alpha, adaptive=True)

    def test_calibration(self):
        alpha = 0.3
        null = 'shoulder'
        # compute_calibration(self.calibration_file, 'dip', null, alpha, adaptive=True)
            # Computing calibration takes too long, so here we will just artificially save some values
        save_lambda(1.3, 'dip_ad', null, alpha, upper=True, lambda_file=self.calibration_file)
        save_lambda(1.2, 'dip_ad', null, alpha, upper=False, lambda_file=self.calibration_file)
        if self.rank == 0:
            print_computed_calibration(self.calibration_file)
        if self.rank == 0:
            data = np.random.randn(1000)
        else:
            data = None
        data = self.comm.bcast(data)
        calibrated_diptest(data, alpha, null, calibration_file=self.calibration_file)

    def tearDown(self):
        if self.rank == 0:
            os.remove(self.calibration_file)


if __name__ == '__main__':
    unittest.main()
