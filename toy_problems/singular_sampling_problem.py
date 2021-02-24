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
D = 5

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


class Q_1D(Q_Base):
    """ Define general (1D) proposal """

    def logpdf(self, x, x_cond):
        return  -0.5 * (x - x_cond)**2

    def rvs(self, x_cond):
        return x_cond + np.random.randn(1)


class L_1D(L_Base):
    """ Define (1D) L-kernel """

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond)**2

class Q(Q_Base):
    """ Define general proposal """
    
    def logpdf(self, x, x_cond):
        return  -0.5 * (x - x_cond).T @ (x - x_cond)
        
    def rvs(self, x_cond):
        return x_cond + np.random.randn(D)


class L(L_Base):
    """ Define L-kernel """
    
    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ (x - x_cond)

p = Target()
q0 = Q0()
q_1d = Q_1D()
l_1d = L_1D()
q = Q()
l = L()

# No. samples and iterations
N = 1000
K = 3

# Standard SMC sampler
smc = SMC(N, D, p, q0, K, q, l)
smc.generate_samples()

# Standard SMC sampler with singular sampling scheme
smc_sin = SMC(N, D, p, q0, K, q_1d, l_1d, sampling='singular')
smc_sin.generate_samples()

# OptL SMC sampler with singular sampling scheme
smc_sin_optL = SMC_OPT(N, D, p, q0, K, q_1d, sampling='singular')
smc_sin_optL.generate_samples()

# Plots of estimated mean
fig, ax = plt.subplots()
ax.plot(np.repeat(2, K), 'lime', linewidth=3.0, label='True value')
for d in range(D):
    ax.plot(smc.mean_estimate_EES[:, d], 'k')
    ax.plot(smc_sin.mean_estimate_EES[:, d], 'r')
    ax.plot(smc_sin_optL.mean_estimate_EES[:, d], 'b')
plt.tight_layout()

# Plot of effective sample size (overview and close-up)
fig, ax = plt.subplots(nrows=2, ncols=1)
for i in range(2):
    ax[i].plot(smc.Neff / smc.N, 'k')
    ax[i].plot(smc_sin.Neff / smc_sin.N, 'r')
    ax[i].plot(smc_sin_optL.Neff / smc_sin_optL.N, 'b')
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('$N_{eff} / N$')
    if i == 0:
        ax[i].set_title('(a)')
        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
    elif i == 1:
        ax[i].set_title('(b)')
        ax[i].set_xlim(0, 20)
    ax[i].set_ylim(0, 1)
##plt.tight_layout()

plt.show()
