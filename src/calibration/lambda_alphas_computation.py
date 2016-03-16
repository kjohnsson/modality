import time

from lambda_alphas_calibration_bw import h_crit_scale_factor

t0 = time.time()

#print "h_crit_scale_factor(0.10) = {}".format(h_crit_scale_factor(0.10))
#print "h_crit_scale_factor(0.05, 'shoulder') = {}".format(h_crit_scale_factor(0.05, 'shoulder'))
print "h_crit_scale_factor(0.05, 'fm', mtol=0.001) = {}".format(h_crit_scale_factor(0.05, 'fm', mtol=0.001))

t1 = time.time()

print "Computation time: {}".format(t1-t0)
