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
Testing for SMC_OPT_GMM

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

    # Define proposal distribution

    def q_mean(x_cond):
        return x_cond

    def q_var(x_cond):
        return 0.1

    q = Normal_PDF_Cond(D=1, mean=q_mean, cov=q_var)

    # Define L-kernel for 'user-defined' implementation

    def L_mean(x_cond):
        return x_cond

    def L_var(x_cond):
        return 0.1

    L = Normal_PDF_Cond(D=1, mean=L_mean, cov=L_var)

    # No. samples and iterations
    N = 5000
    K = 10

    # Run samplers
    smc_opt_gmm = SMC_OPT_GMM(N=N, D=1, p=p, q0=q0, K=K, q=q, L_components=2)
    smc_opt_gmm.generate_samples()

    assert np.allclose(smc_opt_gmm.mean_estimate_EES[-1], 0, atol=0.5)
    assert np.allclose(smc_opt_gmm.var_estimate_EES[-1], 10, atol=0.5)
