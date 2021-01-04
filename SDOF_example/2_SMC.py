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
[x, y_true, y_obs, t, sdof, sigma] = dill.load(file)
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
p = SDOF_Log_Posterior(x, y_obs, q0, sdof)

# No. samples and iterations
N = 500
K = 50

# SMC sampler with user-defined L-kernel
smc = SMC_BASE(N, 3, p, q0, K, q, L)
smc.generate_samples()

# SMC sampler with optimum L-kernel
smc_optL = SMC_OPT(N=N, D=3, p=pi, q0=q0, K=K, q=q)
smc_optL.generate_samples()

# Save results
file_name = 'SMC_results.dat'
file = open(file_name, 'wb')
dill.dump([smc, smc_optL], file)
file.close()