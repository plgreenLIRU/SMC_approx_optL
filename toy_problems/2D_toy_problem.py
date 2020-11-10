import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import *
from SMC_OPT import *

"""
Estimating the optimum L-kernel for a 2D toy problem.

P.L.Green
"""

# Define target distribution
p = Target()
p.pdf = Normal_PDF(mean=np.array([3.0, 2.0]),
                   cov=np.eye(2))
def p_pdf(x):
    return p.pdf.logpdf(x)
p.logpdf = p_pdf

# Define initial proposal
q0 = Q0_Proposal()
q0.pdf = Normal_PDF(mean=np.zeros(2),
                    cov=np.eye(2))
def q0_logpdf(x):
    return q0.pdf.logpdf(x)
def q0_rvs(size):
    return q0.pdf.rvs(size)
q0.logpdf = q0_logpdf
q0.rvs = q0_rvs    

# Define proposal
q = Q_Proposal()
q.pdf = Normal_PDF(cov=np.eye(2))
def q_logpdf(x, x_cond):
    q.pdf.mean = x_cond
    return q.pdf.logpdf(x)
def q_rvs(x_cond):
    q.pdf.mean = x_cond
    return q.pdf.rvs()
q.logpdf = q_logpdf
q.rvs = q_rvs

# Define L-kernel
L = L_Kernel()
L.pdf = Normal_PDF(cov=np.eye(2))
def L_logpdf(x, x_cond):
    L.pdf.mean = x_cond
    return L.pdf.logpdf(x)
L.logpdf = L_logpdf

# No. samples and iterations
N = 500
K = 100

# SMC sampler with user-defined L-kernel
smc = SMC_BASE(N=N, D=2, p=p, q0=q0, K=K, q=q, L=L)
smc.generate_samples()

# SMC sampler with optimum L-kernel
smc_optL = SMC_OPT(N=N, D=2, p=p, q0=q0, K=K, q=q)
smc_optL.generate_samples()

# Print no. of times resampling occurred
print('No. resampling (SMC)', len(smc.resampling_points))
print('No. resampling (SMC optL)', len(smc_optL.resampling_points))

# Print variance of sample estimates
print('\n')
print('E[x1] sample variance: ', np.var(smc.mean_estimate[:, 0]),
      np.var(smc_optL.mean_estimate[:, 0]))
print('E[x2] sample variance: ', np.var(smc.mean_estimate[:, 1]),
      np.var(smc_optL.mean_estimate[:, 1]))
print('Cov[x1,x1] sample variance: ', np.var(smc.var_estimate[:, 0, 0]),
      np.var(smc_optL.var_estimate[:, 0, 0]))
print('Cov[x1,x2] sample variance: ', np.var(smc.var_estimate[:, 0, 1]),
      np.var(smc_optL.var_estimate[:, 0, 1]))
print('Cov[x2,x2] sample variance: ', np.var(smc.var_estimate[:, 1, 1]),
      np.var(smc_optL.var_estimate[:, 1, 1]))

# Plots of estimated mean
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(np.repeat(3, K), 'lime', linewidth=3.0, label='True value')
ax[0].plot(smc.mean_estimate_EES[:, 0], 'k', label='Forward proposal L-kernel')
ax[0].plot(smc_optL.mean_estimate_EES[:, 0], 'r', label='Optimum L-kernel')
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[0].set_title('(a)')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('E[$x_1$]')
ax[1].plot(np.repeat(2, K), 'lime', linewidth=3.0)
ax[1].plot(smc.mean_estimate_EES[:, 1], 'k')
ax[1].plot(smc_optL.mean_estimate_EES[:, 1], 'r')
ax[1].set_title('(b)')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('E[$x_2$]')

plt.tight_layout()

# Plots of estimated elements of covariance matrix
fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(np.repeat(1, K), 'lime', linewidth=3.0, label='True value')
ax[0].plot(smc.var_estimate_EES[:, 0, 0], 'k',
           label='Forward proposal L-kernel')
ax[0].plot(smc_optL.var_estimate_EES[:, 0, 0], 'r', label='Optimum L-kernel')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Cov$[x_1, x_1]$')
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[1].plot(np.repeat(0, K), 'lime', linewidth=3.0)
ax[1].plot(smc.var_estimate_EES[:, 0, 1], 'k')
ax[1].plot(smc_optL.var_estimate_EES[:, 0, 1], 'r')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Cov$[x_1, x_2]$')
ax[2].plot(np.repeat(1, K), 'lime', linewidth=3.0)
ax[2].plot(smc.var_estimate_EES[:, 1, 1], 'k')
ax[2].plot(smc_optL.var_estimate_EES[:, 1, 1], 'r')
ax[2].set_xlabel('Iteration')
ax[2].set_ylabel('Cov$[x_2, x_2]$')

plt.tight_layout()

# Plot of effective sample size (overview and close-up)
fig, ax = plt.subplots(nrows=2, ncols=1)
for i in range(2):
    ax[i].plot(smc.Neff / smc.N, 'k', label='Forward proposal L-kernel')
    ax[i].plot(smc_optL.Neff / smc.N, 'r', label='Optimum L-kernel')
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
