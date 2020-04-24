import numpy as np
import sys
sys.path.append('..')  # noqa
from SMC_BASE import *
from SMC_OPT import *
from scipy.stats import multivariate_normal as Normal_PDF
from Normal_PDF_Cond import *

"""
Testing for SMC_OPT

P.L.Green
"""

# Define target distribution
p = Normal_PDF(mean=np.array([0., 0.]),
               cov=np.array([[1., 0.], [0., 1.]]))

# Define initial proposal
q0 = Normal_PDF(mean=np.array([0., 0.]),
                cov=1.1 * np.array([[1., 0.], [0., 1.]]))

# Define L-kernel


def L_mean(x_cond):
    return x_cond


def L_var(x_cond):
    return 0.01 * np.array([[1, 0], [0, 1]])


L = Normal_PDF_Cond(D=2, mean=L_mean, cov=L_var)

# Define proposal distribution


def q_mean(x_cond):
    return x_cond


def q_var(x_cond):
    return 0.01 * np.array([[1, 0], [0, 1]])


q = Normal_PDF_Cond(D=2, mean=q_mean, cov=q_var)

# No. samples and iterations
N = 1000
K = 20

# SMC sampler with user-defined L-kernel
smc = SMC_BASE(N=N, D=2, p=p, q0=q0, K=K, q=q, L=L)

# SMC sampler with optimum L
smc_opt = SMC_OPT(N=N, D=2, p=p, q0=q0, K=K, q=q)


def test_sampler():
    """ For this simple example, we test that the SMC estimates of target mean
    and variance are reasonably close to the truth.

    """
    # SMC sampler with user defined L
    smc.generate_samples()

    # SMC sampler with approximately optimum L-kernel
    smc_opt.generate_samples()

    # Check estimates
    assert np.allclose(smc_opt.mean_estimate_EES[-1], p.mean, atol=0.1)
    assert np.allclose(smc_opt.var_estimate_EES[-1][0][0], p.cov[0][0],
                       atol=0.2)
    assert np.allclose(smc_opt.var_estimate[-1][1][1], p.cov[1][1],
                       atol=0.2)
    assert np.allclose(smc_opt.var_estimate[-1][0][1], p.cov[0][1],
                       atol=0.2)


def test_Neff():
    """ We'd expect that, on average, our SMC with optimum L-kernel should
    maintain a higher effective sample size that our SMC sampler with a
    'user defined' kernel.

    """

    smc.generate_samples()
    assert np.mean(smc_opt.Neff) > np.mean(smc.Neff)
