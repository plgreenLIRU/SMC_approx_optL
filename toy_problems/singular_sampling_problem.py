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
D = 10

# Variance of general proposal
v = 1

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
        return  -1/(2 * v) * (x - x_cond)**2

    def rvs(self, x_cond):
        return x_cond + np.sqrt(v) * np.random.randn(1)


class L_1D(L_Base):
    """ Define (1D) L-kernel """

    def logpdf(self, x, x_cond):
        return  -1/(2 * v) * (x - x_cond)**2

class Q(Q_Base):
    """ Define general proposal """
    
    def logpdf(self, x, x_cond):
        return  -1/(2 * v) * (x - x_cond).T @ (x - x_cond)
        
    def rvs(self, x_cond):
        return x_cond + np.sqrt(v) * np.random.randn(D)


class L(L_Base):
    """ Define L-kernel """
    
    def logpdf(self, x, x_cond):
        return  -1/(2 * v) * (x - x_cond).T @ (x - x_cond)

p = Target()
q0 = Q0()
q_1d = Q_1D()
l_1d = L_1D()
q = Q()
l = L()

# No. samples and iterations
N = 1000
K = 5

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
fig, ax = plt.subplots(ncols=3)
for i in range(3):
    for d in range(D):
        if i == 0:
            ax[i].plot(smc.mean_estimate_EES[:, d], 'k', 
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_sin.mean_estimate_EES[:, d], 'r', 
                       alpha=0.5)
        if i == 2:
            ax[i].plot(smc_sin_optL.mean_estimate_EES[:, d], 'b', 
                       alpha=0.5)
    ax[i].plot(np.repeat(2, K), 'lime', linewidth=3.0, 
               linestyle='--')
    ax[i].set_ylim([-1, 4])
ax[1].set_xlabel('Iteration')
ax[0].set_ylabel('E[$x$]')
plt.tight_layout()

# Plot of effective sample size (overview and close-up)
fig, ax = plt.subplots()
ax.plot(smc.Neff / smc.N, 'k', 
        label='Forward proposal L-kernel')
ax.plot(smc_sin.Neff / smc_sin.N, 'r', 
        label='Forward proposal L-kernel (Gibbs)')
ax.plot(smc_sin_optL.Neff / smc_sin_optL.N, 'b', 
        label='Optimal L-kernel (Gibbs)')
ax.set_xlabel('Iteration')
ax.set_ylabel('$N_{eff} / N$')
ax.set_ylim([0, 1.1])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.show()
