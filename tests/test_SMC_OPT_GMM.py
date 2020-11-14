import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from GMM_PDF import *
from SMC_BASE import *
from SMC_OPT import *
from SMC_OPT_GMM import *

"""
Estimating optL for a multi-modal toy problem, using a GMM approximate
optimum L-kernel.

P.L.Green
"""


def test_sampler():
    """ Test that we can sample from a multi-modal distribution

    """

    # Define target distribution
    p = GMM_PDF(D=1,
                means=[np.array(-3), np.array(3)],
                vars=[np.array(1), np.array(1)],
                weights=[0.5, 0.5],
                n_components=2)

    # Define initial proposal
    q0 = Normal_PDF(mean=0, cov=3)

    # Define proposal as being Gaussian, centered on x_cond, with variance
    # equal to 0.1
    q = Q_Proposal()
    q.var = 0.1
    q.std = np.sqrt(q.var)
    q.logpdf = lambda x, x_cond : -1/(2*q.var) * (x - x_cond)**2
    q.rvs = lambda x_cond : x_cond + q.std * np.random.randn()

    # Define L-kernel as being Gaussian, centered on x_cond, with variance
    # equal to 0.1
    L = L_Kernel()
    L.var = 0.1
    L.std = np.sqrt(L.var)
    L.logpdf = lambda x, x_cond : -1/(2*L.var) * (x - x_cond)**2

    # No. samples and iterations
    N = 5000
    K = 10

    # Run samplers
    smc_opt_gmm = SMC_OPT_GMM(N=N, D=1, p=p, q0=q0, K=K, q=q, L_components=2)
    smc_opt_gmm.generate_samples()

    assert np.allclose(smc_opt_gmm.mean_estimate_EES[-1], 0, atol=0.5)
    assert np.allclose(smc_opt_gmm.var_estimate_EES[-1], 10, atol=0.5)
