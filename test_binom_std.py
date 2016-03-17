import numpy as np
import matplotlib.pyplot as plt

alpha = 0.05

print "2.58*np.sqrt(alpha*(1-alpha)/18000) = {}".format(2.58*np.sqrt(alpha*(1-alpha)/18000))

n = np.exp(np.linspace(1, 10))

fig, ax = plt.subplots()
ax.scatter(n, 2.58*np.sqrt(alpha*(1-alpha)/n))
ax.set_yscale('log')

plt.show()