import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from Normal_PDF_Cond import *
from GMM_PDF import *
from SMC_BASE import *
from SMC_OPT import *
from SMC_OPT_GMM import *

"""
Estimating optL for a multi-modal toy problem, using a GMM approximate
optimum L-kernel.

P.L.Green
"""

# Define target distribution
p = GMM_PDF(D=1,
            means=[np.array(-3), np.array(3)],
            vars=[np.array(1), np.array(1)],
            weights=[0.5, 0.5],
            n_components=2)

# Define initial proposal
q0 = Normal_PDF(mean=0, cov=3)

# Define proposal distribution
q_mean = lambda x_cond: x_cond
q_var = lambda x_cond: 0.1
q = Normal_PDF_Cond(D=1, mean=q_mean, cov=q_var)

# Define L-kernel for 'user-defined' implementation
L_mean = lambda x_cond: x_cond
L_var = lambda x_cond: 0.1
L = Normal_PDF_Cond(D=1, mean=L_mean, cov=L_var)

# No. samples and iterations
N = 500
K = 1000

# SMC samplers
smc = SMC_BASE(N=N, D=1, p=p, q0=q0, K=K, q=q, L=L)
smc_opt = SMC_OPT(N=N, D=1, p=p, q0=q0, K=K, q=q)
smc_opt_gmm = SMC_OPT_GMM(N=N, D=1, p=p, q0=q0, K=K, q=q, L_components=2)
smc.generate_samples()
smc_opt.generate_samples()
smc_opt_gmm.generate_samples()

# Print no. of times resampling occurred
print('No. resampling (SMC)', len(smc.resampling_points))
print('No. resampling (SMC optL)', len(smc_opt.resampling_points))
print('No. resampling (SMC optL gmm)', len(smc_opt_gmm.resampling_points))

# Print variance of sample estimates
print('\n')
print('E[x] sample variance: ', 
      np.var(smc.mean_estimate), 
      np.var(smc_opt.mean_estimate), 
      np.var(smc_opt_gmm.mean_estimate))
print('V[x] sample variance: ', 
      np.var(smc.var_estimate),
      np.var(smc_opt.var_estimate), 
      np.var(smc_opt_gmm.var_estimate))

# Start plots
fig, ax = plt.subplots(ncols=1, nrows=2)

# Plots of estimated mean
ax[0].plot(np.repeat(0, K), 'lime', linewidth=3.0, label='True value')
ax[0].plot(smc.mean_estimate_EES, 'k', label='Forward proposal L-kernel')
ax[0].plot(smc_opt.mean_estimate_EES, 'r',
           label='Optimum L-kernel \n (1 component)')
ax[0].plot(smc_opt_gmm.mean_estimate_EES, 'b',
           label='Optimum L-kernel \n (2 components)')
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('E[$x$]')

# Plots of estimated variance
ax[1].plot(np.repeat(10, K), 'lime', linewidth=3.0, label='True value')
ax[1].plot(smc.var_estimate_EES, 'k', label='Forward proposal L-kernel')
ax[1].plot(smc_opt.var_estimate_EES, 'r',
           label='Optimum L-kernel \n (1 component)')
ax[1].plot(smc_opt_gmm.var_estimate_EES, 'b',
           label='Optimum L-kernel \n (2 components)')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Var[$x$]')
plt.tight_layout()
fig.savefig('../notes/figures/2_component_mean_var.pdf')

# Plot of effective sample size (overview and close-up)
fig, ax = plt.subplots(nrows=2, ncols=1)
for i in range(2):
    ax[i].plot(smc.Neff / smc.N, 'k', label='Forward proposal L-kernel')
    ax[i].plot(smc_opt.Neff / smc_opt.N, 'r',
               label='Optimum L-kernel \n (1 component)')
    ax[i].plot(smc_opt_gmm.Neff / smc_opt_gmm.N, 'b',
               label='Optimum L-kernel \n (2 components)')
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('$N_{eff} / N$')
    if i == 0:
        ax[i].set_title('(a)')
        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
    elif i == 1:
        ax[i].set_title('(b)')
        ax[i].set_xlim(0, 50)
    ax[i].set_ylim(0, 1)
plt.tight_layout()
fig.savefig('../notes/figures/2_component_Neff.pdf')

plt.show()
