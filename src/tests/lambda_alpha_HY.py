# lambda_alpha according to Hall & York 2001.

import numpy as np
import matplotlib.pyplot as plt

a1 = 0.94029
a2 = -1.59914
a3 = 0.17695
a4 = 0.48971
a5 = -1.77793
a6 = 0.36162
a7 = 0.42423

lambda_al = lambda alpha: (a1*alpha**3 + a2*alpha**2 + a3*alpha + a4)/(alpha**3 + a5*alpha**2 + a6*alpha + a7)

print "lambda_al(0.05) = {}".format(lambda_al(0.05))

alpha = np.linspace(0, 1)
plt.plot(alpha, lambda_al(alpha))
plt.scatter(0.05, 1.1273, marker='+', color='red')
plt.show()