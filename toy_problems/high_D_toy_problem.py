import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import Target_Base, Q0_Base, Q_Base, L_Base
from SMC_OPT import *

"""
Estimating the optimum L-kernel for a D-dimensional toy problem using the
single_step proposal approach.

P.L.Green
"""

# Dimension of problem
D = 10

class Target(Target_Base):
    """ Define target """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.repeat(2, D), cov=0.1*np.eye(D))

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
        return  -0.5 * (x - x_cond)**2

class Q(Q_Base):
    """ Define general proposal """

    def logpdf(self, x, x_cond):
        return  -0.5 * (x - x_cond).T @ (x - x_cond)

    def rvs(self, x_cond):
        return x_cond + np.random.randn(D)

p = Target()
q0 = Q0()
q_1d = Q_1D()
l_1d = L_1D()
q = Q()

# No. samples and iterations
N = 5000
K = 10

# OptL SMC sampler with batch sampling scheme
smc_optL = SMC_OPT(N, D, p, q0, K, q, sampling='batch')
smc_optL.generate_samples()

# OptL SMC sampler with single_step sampling scheme
smc_gib_optL = SMC_OPT(N, D, p, q0, K, q_1d, sampling='single_step')
smc_gib_optL.generate_samples()

# Plots of estimated mean
fig, ax = plt.subplots(ncols=2)
for i in range(2):
    for d in range(D):
        if i == 0:
            ax[i].plot(smc_optL.mean_estimate_EES[:, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_gib_optL.mean_estimate_EES[:, d], 'r',
                       alpha=0.5)
    ax[i].plot(np.repeat(2, K), 'lime', linewidth=3.0,
               linestyle='--')
    ax[i].set_ylim([-2, 5])
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('E[$x$]')
    if i == 0:
        ax[i].set_title('(a)')
    if i == 1:
        ax[i].set_title('(b)')
plt.tight_layout()

# Plots of estimated diagonal elements of covariance matrix
fig, ax = plt.subplots(ncols=2)
for i in range(2):
    for d in range(D):
        if i == 0:
            ax[i].plot(smc_optL.var_estimate_EES[:, d, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_gib_optL.var_estimate_EES[:, d, d], 'r',
                       alpha=0.5)
    ax[i].plot(np.repeat(0.1, K), 'lime', linewidth=3.0,
               linestyle='--')
    ax[i].set_ylim([0, 0.5])
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('Var[$x$]')
    if i == 0:
        ax[i].set_title('(a)')
    if i == 1:
        ax[i].set_title('(b)')
plt.tight_layout()

# Plot of effective sample size
fig, ax = plt.subplots()
ax.plot(smc_optL.Neff / smc_optL.N, 'k',
        label='Optimal L-kernel (batch)')
ax.plot(smc_gib_optL.Neff / smc_gib_optL.N, 'r',
        label='Optimal L-kernel (single step)')
ax.set_xlabel('Iteration')
ax.set_ylabel('$N_{eff} / N$')
ax.set_ylim([0, 1.1])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.show()
