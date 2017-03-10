import unittest
import numpy as np

from modality import calibrated_diptest_fc, calibrated_bwtest_fc, \
    silverman_bwtest_fc, hartigan_diptest_fc, excess_mass_modes_fc,\
    preprocess_fcdata, infer_blur_delta


class testFcInterface(unittest.TestCase):

    def setUp(self):
        self.data = np.random.randn(100)
        self.data_trunc = np.round(100*self.data)*.01
        self.data_trunc[self.data_trunc < 0] = 0
        self.ch_range = 0, np.max(self.data_trunc)
        self.blur_delta = infer_blur_delta(self.data_trunc)
        self.alpha = 0.05

    def test_fc_preprocessing(self):
        calibrated_diptest_fc(self.data, self.alpha, 'shoulder', blurring='none', rm_extreme=False)
        calibrated_diptest_fc(self.data_trunc, self.alpha, 'shoulder', blurring='fp')
        calibrated_diptest_fc(self.data_trunc, self.alpha, 'shoulder', blurring='fp_deterministic')
        calibrated_diptest_fc(self.data_trunc, self.alpha, 'shoulder', blurring='standard')

    def test_fc_separate_preprocessing(self):
        data_preproc = preprocess_fcdata(self.data_trunc, blurring='fp')
        calibrated_diptest_fc(data_preproc, self.alpha, 'shoulder', blurring='none', rm_extreme=False)
        calibrated_bwtest_fc(data_preproc, self.alpha, 'shoulder', blurring='none', rm_extreme=False)
        hartigan_diptest_fc(data_preproc, blurring='none', rm_extreme=False)
        silverman_bwtest_fc(data_preproc, self.alpha, blurring='none', rm_extreme=False)
        excess_mass_modes_fc(data_preproc, blurring='none', rm_extreme=False)

    def test_fc_separate_param(self):
        calibrated_diptest_fc(self.data, self.alpha, 'shoulder', ch_range=self.ch_range, blur_delta=self.blur_delta)
        self.assertRaises(TypeError, calibrated_bwtest_fc, self.data_trunc, self.alpha, 'shoulder', 'auto')
        calibrated_bwtest_fc(self.data_trunc, self.alpha, 'shoulder', I='auto', ch_range=self.ch_range, blur_delta=self.blur_delta)

    def test_fc_no_truncation(self):
        self.assertRaises(ValueError, calibrated_bwtest_fc, self.data, self.alpha, 'shoulder')


if __name__ == '__main__':
    unittest.main()