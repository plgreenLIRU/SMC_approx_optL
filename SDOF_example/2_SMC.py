from SDOF_Class import SDOF
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append('..')  # noqa
import dill
from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from SDOF_Log_Posterior_Class import SDOF_Log_Posterior
from SMC_BASE import SMC, Q0_Base, Q_Base, L_Base
from SMC_OPT import *

# Load training data and model
file_name = 'training_data.dat'
file = open(file_name, 'rb')
[F, y_true, y_obs, t, sdof, sigma] = dill.load(file)
file.close()


class Q0(Q0_Base):
    """ Define prior """

    def __init__(self):
        """ Define prior pdfs over stiffness, damping, and
            noise std.
        """

        self.p_k = Normal_PDF(mean=3, cov=0.5)
        self.p_c = Gamma_PDF(a=1, scale=0.1)
        self.p_sigma = Gamma_PDF(a=1, scale=0.1)

    def logpdf(self, x):

        # Convert to 2D array if currently 1D
        if len(np.shape(x))==1:
            x = np.array([x])

        # Calculate logpdf
        logpdf = (self.p_k.logpdf(x[:, 0]) +
                  self.p_c.logpdf(x[:, 1]) +
                  self.p_sigma.logpdf(x[:, 2]))

        return logpdf

    def rvs(self, size):

        k = np.vstack(self.p_k.rvs(size))
        c = np.vstack(self.p_c.rvs(size))
        sigma = np.vstack(self.p_sigma.rvs(size))

        return np.hstack([k, c, sigma])


class Q(Q_Base):
    """ Define general proposal """

    def __init__(self):
        self.cov = 0.01 * np.array([[0.01, 0, 0],
                                    [0, 0.001, 0],
                                    [0, 0, 0.001]])
        self.inv_cov = np.linalg.inv(self.cov)

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ self.inv_cov @ (x - x_cond)
        
    def rvs(self, x_cond):
        return (x_cond +
                np.sqrt(np.diag(self.cov)) * np.random.randn(3))


class L(L_Base):
    """ Define L-kernel """

    def __init__(self):
        cov = 0.01 * np.array([[0.01, 0, 0],
                               [0, 0.001, 0],
                               [0, 0, 0.001]])
        self.inv_cov = np.linalg.inv(cov)

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ self.inv_cov @ (x - x_cond)


# Define log target distribution
p = SDOF_Log_Posterior(F, y_obs, Q0(), sdof)

# No. samples and iterations
N = 500
K = 50

# SMC sampler with user-defined L-kernel
smc = SMC(N, 3, p, Q0(), K, Q(), L())
smc.generate_samples()

# SMC sampler with optimum L-kernel
smc_optL = SMC_OPT(N, 3, p, Q0(), K, Q())
smc_optL.generate_samples()

# Plot SMC results (estimates of means)
fig, ax = plt.subplots(nrows=3)
for i in range(3):
    ax[i].plot(smc.mean_estimate_EES[:, i], 'k',
               label='Forward proposal L-kernel')
    ax[i].plot(smc_optL.mean_estimate_EES[:, i], 'r',
               label='Optimal L-kernel')
    ax[i].set_xlabel('Iteration')
    if i == 0:
        ax[i].set_ylabel('E[$k$]')
    if i == 1:
        ax[i].set_ylabel('E[$c$]')
    if i == 2:
        ax[i].set_ylabel('E[$\sigma$]')

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Plot SMC results (estimates of covariance terms)
fig, ax = plt.subplots(nrows=3)
for i in range(3):
    ax[i].plot(smc.var_estimate_EES[:, i, i], 'k',
               label='Forward proposal L-kernel')
    ax[i].plot(smc_optL.var_estimate_EES[:, i, i], 'r',
               label='Optimal L-kernel')
    ax[i].set_xlabel('Iteration')
    if i == 0:
        ax[i].set_ylabel('Var[$k$]')
    if i == 1:
        ax[i].set_ylabel('Var[$c$]')
    if i == 2:
        ax[i].set_ylabel('Var[$\sigma$]')

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Plot of effective sample size (overview and close-up)
fig, ax = plt.subplots(nrows=2, ncols=1)
for i in range(2):
    ax[i].plot(smc.Neff / smc.N, 'k', label='Forward proposal L-kernel')
    ax[i].plot(smc_optL.Neff / smc.N, 'r', label='Optimal L-kernel')
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('$N_{eff} / N$')
    if i == 0:
        ax[i].set_title('(a)')
        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
    elif i == 1:
        ax[i].set_title('(b)')
        ax[i].set_xlim(0, 20)
    ax[i].set_ylim(0, 1)
plt.tight_layout()

plt.show()
