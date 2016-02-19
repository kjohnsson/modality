from scipy.special import beta as betafun
import numpy as np
import cPickle as pickle

beta_betadistr = np.exp(np.arange(20))
gamma_betadistr = lambda beta: 2*(beta-1)*betafun(beta, 1.0/2)**2
ref_betadistr = gamma_betadistr(beta_betadistr)
print "gamma_betadistr(beta_betadistr) = {}".format(gamma_betadistr(beta_betadistr))
print "ref_betadistr[-1]-2*np.pi = {}".format(ref_betadistr[-1]-2*np.pi)

beta_studentt = 1./2*np.exp(np.arange(20))
gamma_studentt = lambda beta: 2*beta*betafun(beta-1./2, 1./2)**2
ref_studentt = gamma_studentt(beta_studentt)
print "gamma_studentt(beta_studentt) = {}".format(gamma_studentt(beta_studentt))
print "ref_studentt[-1]-2*np.pi = {}".format(ref_studentt[-1]-2*np.pi)

savedat = {'beta_betadistr': beta_betadistr,
           'gamma_betadistr': gamma_betadistr(beta_betadistr),
           'beta_studentt': beta_studentt,
           'gamma_studentt': gamma_studentt(beta_studentt)}

with open('gammaval.pkl', 'w') as f:
    pickle.dump(savedat, f, -1)

