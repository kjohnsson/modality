import unittest
import numpy as np
from sklearn.neighbors import KernelDensity
import time

from modality.util import ApproxGaussianKDE, auto_interval, fp_blurring


class TestUtil(unittest.TestCase):

    def test_KDE(self):

        for N in [10, 100, 1000, 10000, 100000]:
            h = 1
            data = np.random.randn(N)
            kde = ApproxGaussianKDE(data, h)
            KD = KernelDensity(kernel='gaussian', bandwidth=h, rtol=1e-4).fit(data.reshape(-1, 1))
            x = np.linspace(-3, 3, 400)
            xresh = x.reshape(-1, 1)
            t0 = time.time()
            y_apr_prop = kde.evaluate_prop(x)
            t1 = time.time()
            y_KD = np.exp(KD.score_samples(xresh))
            t2 = time.time()

            y_apr = y_apr_prop / kde._norm_factor

            #print "np.max(np.abs(y_apr-y_KD)) = {}".format(np.max(np.abs(y_apr-y_KD)))
            self.assertTrue(np.allclose(y_apr, y_KD, atol=1e-5))

            print "Speedup for approx KDE vs. KernelDensity with {} data points: {}".format(N, (t2-t1)/(t1-t0))

    def test_auto_interval(self):
        data = np.random.randn(1000)
        auto_interval(data, xmin=np.min(data), xmax=np.max(data), dx=(np.max(data)-np.min(data))/1000)

    def test_fp_blurring(self):
        data = 7*np.random.randn(1000)
        data_trunc = np.round(data)
        w = 1.0
        data_blurred = fp_blurring(data_trunc, w)
        self.assertTrue(np.max(np.abs(data_blurred-data_trunc)) <= 0.5*w)

    def test_fp_blurring_deterministic(self):
        data = 7*np.random.randn(1000)
        data_trunc = np.round(data)
        w = 1.0
        data_blurred = fp_blurring(data_trunc, w, even_spaced=True)
        self.assertTrue(np.max(np.abs(data_blurred-data_trunc)) <= 0.5*w)


if __name__ == '__main__':
    unittest.main()