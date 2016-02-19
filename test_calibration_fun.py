from scipy.special import beta as betafun
from scipy.special import psi
from scipy.optimize import newton, brentq
import matplotlib.pyplot as plt
import numpy as np
import time
import cPickle as pickle

with open('data/gammaval.pkl', 'r') as f:
    savedat = pickle.load(f)

if 0:
    beta = np.linspace(1, 200, 200)
    #gamma = np.abs(2*betafun(beta, beta)**2*(beta-1)*4**(2*beta+5))
    gamma = 2*(beta-1)*(betafun(beta, beta)*2**(2*beta-1))**2
    gamma2 = 2*(beta-1)*betafun(beta, 1.0/2)**2
    print "gamma[-1] = {}".format(gamma[-1])
    print "gamma2[-1] = {}".format(gamma2[-1])
    print "2*np.pi = {}".format(2*np.pi)

    gamma_deriv = 2*betafun(beta, 1./2)**2*(1+2*(beta-1)*(psi(beta)-psi(1./2+beta)))

    #plt.plot(beta, betafun(beta, beta))
    plt.plot(beta, gamma)
    plt.plot(beta, gamma2)
    plt.plot(savedat['beta_betadistr'][:6], savedat['gamma_betadistr'][:6])

    plt.figure()
    plt.semilogy(beta, gamma_deriv)
    plt.semilogy(beta[1:], np.diff(gamma))

    plt.show()

if 1:
    d = 2*np.pi-1e-4
    gamma = lambda beta: 2*(beta-1)*betafun(beta, 1.0/2)**2 - d
    gamma_deriv = lambda beta: 2*betafun(beta, 1./2)**2*(1+2*(beta-1)*(psi(beta)-psi(1./2+beta)))

    beta_init = 1.5

    i = np.searchsorted(savedat['gamma_betadistr'], d)
    beta_left = savedat['beta_betadistr'][i-1]
    beta_right = savedat['beta_betadistr'][i]
    print "brentq(gamma, beta_left, beta_right) = {}".format(brentq(gamma, beta_left, beta_right))
    #print "gamma(47123.7421807) = {}".format(gamma(47123.7421807))

    #print "newton(gamma, beta_init, gamma_deriv) = {}".format(newton(gamma, beta_init, gamma_deriv))
    print "newton(gamma, beta_init) = {}".format(newton(gamma, beta_init))
    #print "gamma(47123.9421526) = {}".format(gamma(47123.9421526))
    #print "brentq(gamma, 0, 2*np.pi) = {}".format(brentq(gamma, 0, 20))

    t0 = time.time()

    #for k in range(100):
    #    newton(gamma, beta_init, gamma_deriv)

    t1 = time.time()

    for k in range(100):
        newton(gamma, beta_init)

    t2 = time.time()

    # for k in range(100):
    #     brentq(gamma, 0, 20)

    t3 = time.time()

    for k in range(100):
        i = np.searchsorted(savedat['gamma_betadistr'], d)
        beta_left = savedat['beta_betadistr'][i-1]
        beta_right = savedat['beta_betadistr'][i]
        brentq(gamma, beta_left, beta_right)

    t4 = time.time()

    print "solve with derivative = {}".format(t1-t0)
    print "solve without derivative = {}".format(t2-t1)
    print "solve with brentq = {}".format(t3-t2)
    print "solve with saved values = {}".format(t4-t3)

if 0:
    beta = np.linspace(1./2, 200, 200)
    gamma = 2*beta*betafun(beta-1./2, 1./2)**2
    print "gamma[-1] = {}".format(gamma[-1])
    print "2*np.pi = {}".format(2*np.pi)

    gamma_deriv = 2*betafun(beta-1./2, 1./2)**2*(1+2*beta*(psi(beta-1./2)-psi(beta)))

    plt.plot(beta, gamma)
    plt.plot(savedat['beta_studentt'][:7], savedat['gamma_studentt'][:7])

    plt.figure()
    plt.semilogy(beta, -gamma_deriv)
    plt.semilogy(beta[1:], -np.diff(gamma))

    plt.show()

if 0:
    d = 2*np.pi+1e-4  # 20
    gamma = lambda beta: 2*beta*betafun(beta-1./2, 1./2)**2 - d
    gamma_deriv = lambda beta: 2*betafun(beta-1./2, 1./2)**2*(1+2*beta*(psi(beta-1./2)-psi(beta)))

    beta_init = 1.0  #1.5

    i = np.searchsorted(-savedat['gamma_studentt'], -d)
    beta_left = savedat['beta_studentt'][i-1]
    beta_right = savedat['beta_studentt'][i]
    print "brentq(gamma, beta_left, beta_right) = {}".format(brentq(gamma, beta_left, beta_right))

    #print "newton(gamma, beta_init, gamma_deriv) = {}".format(newton(gamma, beta_init, gamma_deriv))
    print "newton(gamma, beta_init) = {}".format(newton(gamma, beta_init))
    print "brentq(gamma, 1./2, 1) = {}".format(brentq(gamma, 1./2, 1))

    t0 = time.time()

    for k in range(100):
        newton(gamma, beta_init, gamma_deriv)

    t1 = time.time()

    for k in range(100):
        newton(gamma, beta_init)

    t2 = time.time()

    for k in range(100):
        brentq(gamma, 1./2, 200)

    t3 = time.time()

    for k in range(100):
        i = np.searchsorted(-savedat['gamma_studentt'], -d)
        beta_left = savedat['beta_studentt'][i-1]
        beta_right = savedat['beta_studentt'][i]
        brentq(gamma, beta_left, beta_right)

    t4 = time.time()

    print "solve with derivative = {}".format(t1-t0)
    print "solve without derivative = {}".format(t2-t1)
    print "solve with brentq = {}".format(t3-t2)
    print "solve with saved values = {}".format(t4-t3)

if 0:
    d = 2*np.pi+1e-3 # 2*np.pi+1e-2
    gamma = lambda beta: 2*beta*betafun(beta-1./2, 1./2)**2 - d
    gamma_deriv = lambda beta: 2*betafun(beta-1./2, 1./2)**2*(1+2*beta*(psi(beta-1./2)-psi(beta)))

    if d < 2*np.pi+1e-2:
        beta_init = 1000
    elif d < 7:
        beta_init = 10
    else:
        beta_init = 1

    #print "newton(gamma, beta_init, gamma_deriv) = {}".format(newton(gamma, beta_init, gamma_deriv))
    print "newton(gamma, beta_init) = {}".format(newton(gamma, beta_init))
    #print "brentq(gamma, 1./2, 1) = {}".format(brentq(gamma, 1./2, 1))

    t0 = time.time()

    for k in range(100):
        newton(gamma, beta_init, gamma_deriv)

    t1 = time.time()

    for k in range(100):
        newton(gamma, beta_init)

    t2 = time.time()

    #for k in range(100):
    #    brentq(gamma, 1./2, 200)

    #t3 = time.time()

    print "solve with derivative = {}".format(t1-t0)
    print "solve without derivative = {}".format(t2-t1)
    #print "solve with brentq = {}".format(t3-t2)