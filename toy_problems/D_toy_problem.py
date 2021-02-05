import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import *
from SMC_OPT import *

"""
Estimating the optimum L-kernel for a D-dimensional toy problem.

P.L.Green
"""
np.random.seed(10)
# Dimension of problem
D = 4

# Define target distribution 
p = Normal_PDF(mean=np.repeat(1, D), cov=np.eye(D))

# Define initial proposal
q0 = Normal_PDF(mean=np.zeros(D), cov=np.eye(D))  

# Define proposal as being Gaussian, centered on x_cond
q = Q_Proposal()
q.logpdf = lambda x, x_cond : -0.5 * (x - x_cond).T @ (x - x_cond)
q.rvs = lambda x_cond : x_cond + 1 * np.random.randn(D)

# Define L-kernel as being Gaussian, centered on x_cond
L = L_Kernel()
L.logpdf = lambda x, x_cond : -0.5 * (x - x_cond).T @ (x - x_cond)
L.rvs = q.rvs

# No. samples and iterations
N = 1000
K = 50

# SMC sampler with user-defined L-kernel
smc = SMC_BASE(N=N, D=D, p=p, q0=q0, K=K, q=q, L=L)
smc.generate_samples()

# SMC sampler with optimum L-kernel
smc_optL = SMC_OPT(N=N, D=D, p=p, q0=q0, K=K, q=q)
smc_optL.generate_samples()

# SMC sampler with optimum L-kernel QR implementation 
smc_optL_qr = SMC_OPT(N=N, D=D, p=p, q0=q0, K=K, q=q, PCA = 'naive', t = 4)
smc_optL_qr.generate_samples()

# SMC sampler with optimum L-kernel eigh implementation 
smc_optL_eigh = SMC_OPT(N=N, D=D, p=p, q0=q0, K=K, q=q, PCA = 'eigh', t = 4)
smc_optL_eigh.generate_samples()

# SMC sampler with optimum L-kernel PCA implementation 
smc_optL_svd = SMC_OPT(N=N, D=D, p=p, q0=q0, K=K, q=q, PCA = 'svd', t = 4)
smc_optL_svd.generate_samples()

# Print no. of times resampling occurred
print('No. resampling (SMC)', len(smc.resampling_points))
print('No. resampling (SMC optL)', len(smc_optL.resampling_points))
print('No. resampling (SMC optL (QR))', len(smc_optL_qr.resampling_points))
print('No. resampling (SMC optL (eigh))', len(smc_optL_eigh.resampling_points))
print('No. resampling (SMC optL (PCA))', len(smc_optL_svd.resampling_points))

# Plots of estimated means
fig, ax = plt.subplots(nrows=D, ncols=1)
fig.subplots_adjust(hspace=0.0)
for i in range(D):
    ax[i].plot(np.repeat(1, K), 'lime', linewidth=3.0, 
               label='True value')
    ax[i].plot(smc.mean_estimate_EES[:, i], 'k', 
               label='Forward proposal L-kernel')
    ax[i].plot(smc_optL.mean_estimate_EES[:, i], 'r', 
               label='Optimum L-kernel')
    ax[i].plot(smc_optL_qr.mean_estimate_EES[:, i], 'b', 
               label='Optimum L-kernel (QR)')
    ax[i].plot(smc_optL_eigh.mean_estimate_EES[:, i], 'm--', 
               label='Optimum L-kernel (eigh)')
    ax[i].plot(smc_optL_svd.mean_estimate_EES[:, i], 'g--', 
               label='Optimum L-kernel (PCA)')
    if i == 0:
        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()

# Plot of effective sample size (overview and close-up)
fig, ax = plt.subplots(nrows=2, ncols=1)
for i in range(2):
    ax[i].plot(smc.Neff / smc.N, 'k', label='Forward proposal L-kernel')
    ax[i].plot(smc_optL.Neff / smc.N, 'r', label='Optimum L-kernel')
    ax[i].plot(smc_optL_qr.Neff / smc.N, 'b', label='Optimum L-kernel (QR)')
    ax[i].plot(smc_optL_eigh.Neff / smc.N, 'm--', label='Optimum L-kernel (eigh)')
    ax[i].plot(smc_optL_svd.Neff / smc.N, 'g--', label='Optimum L-kernel (PCA)')
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
