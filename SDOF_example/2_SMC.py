from SDOF_Class import SDOF
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append('..')  # noqa
import dill
from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from SDOF_Log_Posterior_Class import SDOF_Log_Posterior
from SMC_BASE import *
from SMC_OPT import *

# Load training data and model
file_name = 'training_data.dat'
file = open(file_name, 'rb')
[F, y_true, y_obs, t, sdof, sigma] = dill.load(file)
file.close()

# Define prior
q0 = Q0_Proposal()
q0.p_k = Normal_PDF(mean=3, cov=0.5)
q0.p_c = Gamma_PDF(a=1, scale=0.1)
q0.p_sigma = Gamma_PDF(a=1, scale=0.1)

def q0_rvs(size):
    k = np.vstack(q0.p_k.rvs(size))
    c = np.vstack(q0.p_c.rvs(size))
    sigma = np.vstack(q0.p_sigma.rvs(size))

    return np.hstack([k, c, sigma])

def q0_logpdf(x):

    # Convert to 2D array if currently 1D
    if len(np.shape(x))==1:
        x = np.array([x])

    # Calculate logpdf
    logpdf = (q0.p_k.logpdf(x[:, 0]) +
              q0.p_c.logpdf(x[:, 1]) +
              q0.p_sigma.logpdf(x[:, 2]))

    return logpdf

q0.rvs = q0_rvs
q0.logpdf = q0_logpdf

# Define L-kernel
L = L_Kernel()
L.cov = 0.01 * np.array([[0.01, 0, 0],
                         [0, 0.001, 0],
                         [0, 0, 0.001]])
L.inv_cov = np.linalg.inv(L.cov)
L.logpdf = lambda x, x_cond : (-0.5 * (x - x_cond).T @
                               L.inv_cov @ (x - x_cond))

# Define proposal
q = Q_Proposal()
q.cov = 0.01 * np.array([[0.01, 0, 0],
                         [0, 0.001, 0],
                         [0, 0, 0.001]])
q.inv_cov = np.linalg.inv(q.cov)
q.logpdf = lambda x, x_cond : (-0.5 * (x - x_cond).T @
                               q.inv_cov @ (x - x_cond))
q.rvs = lambda x_cond : (x_cond +
                         np.sqrt(np.diag(q.cov)) * np.random.randn(3))

# Define log target distribution
p = SDOF_Log_Posterior(F, y_obs, q0, sdof)

# No. samples and iterations
N = 500
K = 50

# SMC sampler with user-defined L-kernel
smc = SMC_BASE(N, 3, p, q0, K, q, L)
smc.generate_samples()

# SMC sampler with optimum L-kernel
smc_optL = SMC_OPT(N, 3, p, q0, K, q)
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
