import numpy as np
import sys
sys.path.append('..')  # noqa
from SMC_BASE import *
from scipy.stats import multivariate_normal as Normal_PDF

"""
Testing for SMC_BASE

P.L.Green
"""

# Define target distribution 
p = Normal_PDF(mean=np.array([3.0, 2.0]), cov=np.eye(2))

# Define initial proposal
q0 = Normal_PDF(mean=np.zeros(2), cov=np.eye(2))  

# Define proposal as being Gaussian, centered on x_cond, with identity 
# covariance matrix
q = Q_Proposal()
q.logpdf = lambda x, x_cond : -0.5 * (x - x_cond).T @ (x - x_cond)
q.rvs = lambda x_cond : x_cond + np.random.randn(2)

# Define L-kernel as being Gaussian, centered on x_cond, with identity 
# covariance matrix
L = L_Kernel()
L.logpdf = lambda x, x_cond : -0.5 * (x - x_cond).T @ (x - x_cond)
L.rvs = lambda x_cond : x_cond + np.random.randn(2)

# No. samples and iterations
N = 1000
K = 500

# SMC sampler with user-defined L-kernel
smc = SMC_BASE(N=N, D=2, p=p, q0=q0, K=K, q=q, L=L)


def test_sampler():
    """ For this simple example, we test that the SMC estimates of target mean
    and variance are reasonably close to the truth.

    """

    # SMC sampler with user-defined L-kernel
    smc.generate_samples()

    # Check estimates
    assert np.allclose(smc.mean_estimate_EES[-1], p.mean, atol=0.1)
    assert np.allclose(smc.var_estimate_EES[-1][0][0], p.cov[0][0],
                       atol=0.2)
    assert np.allclose(smc.var_estimate[-1][1][1], p.cov[1][1],
                       atol=0.2)
    assert np.allclose(smc.var_estimate[-1][0][1], p.cov[0][1],
                       atol=0.2)


def test_normalise_weights():
    """ Test that normalised weights always sum to 1 and that we can cope with
    -inf values in the array of low weights.

    """

    logw = np.log(np.random.rand(1, N))
    wn = smc.normalise_weights(logw)
    assert np.allclose(np.sum(wn), 1.0, atol=1e-8)

    logw[0] = -np.inf
    assert np.allclose(np.sum(wn), 1.0, atol=1e-8)
