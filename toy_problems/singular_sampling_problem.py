import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import SMC, Target_Base, Q0_Base, Q_Base, L_Base
from SMC_OPT import *

"""
Estimating the optimum L-kernel for a D-dimensional toy problem using the
singular sampling approach.

P.L.Green
"""

# Dimension of problem
D = 1

class Target(Target_Base):
    """ Define target """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.repeat(2, D), cov=np.eye(D))

    def logpdf(self, x):
        return self.pdf.logpdf(x)


class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.zeros(D), cov=np.eye(D))


    def logpdf(self, x):
        return self.pdf.logpdf(x)

    def rvs(self, size):
        return self.pdf.rvs(size)


class Q(Q_Base):
    """ Define general (1D) proposal """

    def logpdf(self, x, x_cond):
        return  -0.5 * (x - x_cond)**2

    def rvs(self, x_cond):
        return x_cond + np.random.randn(1)


class L(L_Base):
    """ Define (1D) L-kernel """

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond)**2


p = Target()
q0 = Q0()
q = Q()
l = L()

# No. samples and iterations
N = 1000
K = 500

# Standard SMC sampler
smc = SMC(N, D, p, q0, K, q, l)
smc.generate_samples()

# SMC sampler with singular sampling scheme
smc_sin = SMC(N, D, p, q0, K, q, l, sampling='singular')
smc_sin.generate_samples()

# Plots of estimated mean
fig, ax = plt.subplots()
ax.plot(np.repeat(2, K), 'lime', linewidth=3.0, label='True value')
for d in range(D):
    ax.plot(smc.mean_estimate_EES[:, d], 'k')
    ax.plot(smc_sin.mean_estimate_EES[:, d], 'r')
plt.tight_layout()

plt.show()
