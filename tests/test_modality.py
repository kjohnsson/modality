import unittest
import numpy as np
import time
from mpi4py import MPI

from modality import calibrated_diptest, calibrated_bwtest, silverman_bwtest, hartigan_diptest, excess_mass_modes
from modality.util import auto_interval


class testModality(unittest.TestCase):

    def setUp(self):
        self.data = np.random.randn(1000)
        #self.data = MPI.COMM_WORLD.bcast(self.data)
        self.alpha = 0.05
        self.I = auto_interval(self.data)

    # def test_not_same_data_mpi(self):
    #     if MPI.COMM_WORLD.Get_size() > 1:
    #         self.assertRaises(ValueError, calibrated_diptest,
    #                           np.random.randn(1000), self.alpha, 'normal')

    def test_calibrated_diptest(self):
        t0 = time.time()
        calibrated_diptest(self.data, self.alpha, 'shoulder', adaptive_resampling=True)
        t1 = time.time()
        calibrated_diptest(self.data, self.alpha, 'normal', adaptive_resampling=True)
        t2 = time.time()
        calibrated_diptest(self.data, self.alpha, 'shoulder', adaptive_resampling=False)
        t3 = time.time()
        calibrated_diptest(self.data, self.alpha, 'normal', adaptive_resampling=False)
        t4 = time.time()

        print "Adaptive sampling diptest: {}, {}".format(t1-t0, t2-t1)
        print "Non-adaptive sampling diptest: {}, {}".format(t4-t3, t3-t2)

    def test_calibrated_bwtest(self):
        t0 = time.time()
        calibrated_bwtest(self.data, self.alpha, 'shoulder', self.I, adaptive_resampling=True)
        t1 = time.time()
        calibrated_bwtest(self.data, self.alpha, 'normal', self.I, adaptive_resampling=True)
        t2 = time.time()
        calibrated_bwtest(self.data, self.alpha, 'shoulder', self.I, adaptive_resampling=False)
        t3 = time.time()
        calibrated_bwtest(self.data, self.alpha, 'normal', self.I, adaptive_resampling=False)
        t4 = time.time()

        print "Adaptive sampling bwtest: {}, {}".format(t1-t0, t2-t1)
        print "Non-adaptive sampling bwtest: {}, {}".format(t4-t3, t3-t2)

    def test_silverman_bwtest(self):
        t0 = time.time()
        silverman_bwtest(self.data, self.alpha, self.I, adaptive_resampling=True)
        t1 = time.time()
        silverman_bwtest(self.data, self.alpha, self.I, adaptive_resampling=False)
        t2 = time.time()

        print "Adaptive sampling silverman bwtest: {}".format(t1-t0)
        print "Non-adaptive sampling silverman bwtest: {}".format(t2-t1)

    def test_hartigan_diptest(self):
        t0 = time.time()
        hartigan_diptest(self.data)
        t1 = time.time()

        print "Hartigan diptest: {}".format(t1-t0)

    def test_excess_mass_modes(self):
        t0 = time.time()
        excess_mass_modes(self.data)
        t1 = time.time()

        print "Finding excess mass modes: {}".format(t1-t0)

if __name__ == '__main__':
    unittest.main()