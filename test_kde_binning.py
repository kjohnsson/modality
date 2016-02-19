import numpy as np
import matplotlib.pyplot as plt

if 0:
    x = np.linspace(-2, 2, 200)

    a1 = 1./np.sqrt(2*np.pi)*np.exp(-x**2/2)
    a2 = 1./np.sqrt(2*np.pi)*np.exp(-(x-1./100)**2/2)

    plt.plot(x, a1)
    plt.plot(x, a2)

    plt.figure()
    plt.plot(x, a1-a2)

    plt.show()

if 0:
    alpha = np.sqrt(2e-4)
    x = np.linspace(np.sqrt(2)-alpha, 5, 200)

    a = ((x+alpha)**2-1)*np.exp(-(x-alpha)**2/2)

    plt.plot(x, a)
    plt.show()

h = 1
tol = 1e-4
alpha = np.sqrt(2*tol)
x_i = alpha*h
x = np.linspace(-3, 3, 300)
f = np.exp(-(x-x_i)**2/(2*h**2))
f_macl = np.exp(-x**2/(2*h**2)) + x_i*x/h**2*np.exp(-x**2/(2*h**2))
err_macl = x_i**2/(2*h**4) * h**2*np.ones(len(x))
err_macl2 = x_i**2/(2*h**4)*((x+alpha*h)**2 - h**2)*np.exp(-(x-alpha*h)**2/(2*h**2))
plt.plot(x, f-f_macl)
plt.plot(x, err_macl)
plt.plot(x, err_macl2)
print "np.max(err_macl) = {}".format(np.max(err_macl))
plt.show()
# + \frac{x_i^2}{2h^4} \left((x-\xi_i)^2 - h^2\right) \exp \left(-\frac{(x-\xi_i)^2}{2h^2} \right)
