import numpy as np
import sys
sys.path.append('..')  # noqa
from SMC_BASE import SMC, Target_Base, Q0_Base, Q_Base, L_Base
from SMC_OPT import *
from scipy.stats import multivariate_normal as Normal_PDF

"""
Testing for SMC_OPT

P.L.Green
"""

np.random.seed(42)

class Target(Target_Base):
    """ Define target """

    def __init__(self):
        self.mean = np.array([3.0, 2.0])
        self.cov = np.eye(2)
        self.pdf = Normal_PDF(self.mean, self.cov)

    def logpdf(self, x):
        return self.pdf.logpdf(x)


class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.zeros(2), cov=np.eye(2))


    def logpdf(self, x):
        return self.pdf.logpdf(x)

    def rvs(self, size):
        return self.pdf.rvs(size)


class Q(Q_Base):
    """ Define general proposal """
    
    def logpdf(self, x, x_cond):
        return  -0.5 * (x - x_cond).T @ (x - x_cond)
        
    def rvs(self, x_cond):
        return x_cond + np.random.randn(2)


class L(L_Base):
    """ Define L-kernel """
    
    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ (x - x_cond)

# No. samples and iterations
N = 1000
K = 500

# SMC sampler with user-defined L-kernel
p = Target()
q0 = Q0()
q = Q()
l = L()
smc = SMC(N, 2, p, q0, K, q, l)

# SMC sampler with optimum L
smc_opt = SMC_OPT(N, 2, p, q0, K, q)


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
