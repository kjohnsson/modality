import time

from lambda_alphas_calibration_bw import h_crit_scale_factor, XSampleBW, XSampleShoulderBW

t0 = time.time()

print "h_crit_scale_factor(0.10) = {}".format(h_crit_scale_factor(0.10))

t1 = time.time()

print "Computation time: {}".format(t1-t0)
#print "h_crit_scale_factor(0.05, XSampleShoulderBW) = {}".format(h_crit_scale_factor(0.05, XSampleShoulderBW))