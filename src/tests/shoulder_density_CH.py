import numpy as np
import matplotlib.pyplot as plt

if 0:
    w1 = 16.0
    w2 = 1.0

    m = -1.25
    s = 0.25

###########

if 0:
    w1 = 8*np.exp(9.0/8)
    w2 = 1.0

    m = -9*np.sqrt(3)/8
    s = 0.25

if 1:
    w1 = 100.0
    w2 = 9.0

    m = 1.3
    s = 0.3

w = w1+w2
w1 /= w
w2 /= w
dens = lambda x: w1*np.exp(-x**2/2.0) + w2/s*np.exp(-(x-m)**2/s**2/2.0)

x = np.linspace(-3, 3, 200)
plt.plot(x, dens(x))

plt.figure()
plt.plot(x[1:], np.diff(dens(x)))

plt.show()