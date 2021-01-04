from matplotlib import pyplot as plt
import numpy as np
import dill
import sys
sys.path.append('..')    # noqa
from MH_Class import MH

# Load MCMC results
file_name = 'MCMC_results.dat'
file = open(file_name, 'rb')
mh = dill.load(file)
file.close()

# Load SMC results
file_name = 'SMC_results.dat'
file = open(file_name, 'rb')
[smc, smc_optL] = dill.load(file)
file.close()

# MCMC plot
mean_MCMC, cov_MCMC, fig_MCMC, ax_MCMC = mh.show_results(burnin=200)
ax_MCMC[1][0].set_xlabel('$k$')
ax_MCMC[1][1].set_xlabel('$c$')
ax_MCMC[1][2].set_xlabel('$\sigma$')
ax_MCMC[0][0].set_ylabel('Frequency')
ax_MCMC[0][1].set_ylabel('Frequency')
ax_MCMC[0][2].set_ylabel('Frequency')
ax_MCMC[1][0].set_ylabel('Iteration')
ax_MCMC[1][1].set_ylabel('Iteration')
ax_MCMC[1][2].set_ylabel('Iteration')
plt.tight_layout()

fig_MCMC.savefig('../notes/figures/SDOF_MCMC.pdf')

# Plot SMC results against MCMC results (estimates of means)
fig, ax = plt.subplots(nrows=3)
for i in range(3):
    ax[i].plot(np.array([0, smc_optL.K]), np.repeat(mean_MCMC[i], 2), 
               'lime', linewidth=3.0, linestyle='--', label='MCMC')
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

fig.savefig('../notes/figures/SDOF_mean.pdf')

# Plot SMC results against MCMC results (estimates of variances)
fig, ax = plt.subplots(nrows=3)
for i in range(3):
    ax[i].plot(np.array([0, smc_optL.K]), np.repeat(cov_MCMC[i][i], 2), 
               'lime', linewidth=3.0, linestyle='--', label='MCMC')
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

fig.savefig('../notes/figures/SDOF_var.pdf')

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

fig.savefig('../notes/figures/SDOF_Neff.pdf')

# Print variance of sample estimates
print('\n')
print('E[k] sample variance: ', np.var(smc.mean_estimate[:, 0]),
      np.var(smc_optL.mean_estimate[:, 0]))
print('E[c] sample variance: ', np.var(smc.mean_estimate[:, 1]),
      np.var(smc_optL.mean_estimate[:, 1]))
print('E[sigma] sample variance: ', np.var(smc.mean_estimate[:, 2]),
      np.var(smc_optL.mean_estimate[:, 2]))
print('Var[k] sample variance: ', np.var(smc.var_estimate[:, 0, 0]),
      np.var(smc_optL.var_estimate[:, 0, 0]))
print('Var[c] sample variance: ', np.var(smc.var_estimate[:, 1, 1]),
      np.var(smc_optL.var_estimate[:, 1, 1]))
print('Var[sigma$ sample variance: ', np.var(smc.var_estimate[:, 2, 2]),
      np.var(smc_optL.var_estimate[:, 2, 2]))

plt.show()
