import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from GMM_PDF import *
from SMC_BASE import SMC, Target_Base, Q0_Base, Q_Base, L_Base
from SMC_OPT import *
from SMC_OPT_GMM import *

"""
Estimating optL for a multi-modal toy problem, using a GMM approximate
optimum L-kernel.

P.L.Green
"""

np.random.seed(42)

class Target(Target_Base):
    """ Define target """
    
    def __init__(self):
        self.pdf = GMM_PDF(D=1,
                           means=[np.array(-3), np.array(3)],
                           vars=[np.array(1), np.array(1)],
                           weights=[0.5, 0.5],
                           n_components=2)
                           
    def logpdf(self, x):
        return self.pdf.logpdf(x)


class Q0(Q0_Base):
    """ Define initial proposal """
    
    def __init__(self):
        self.pdf = Normal_PDF(mean=0, cov=3)
        
    def logpdf(self, x):
        return self.pdf.logpdf(x)
        
    def rvs(self, size):
        return self.pdf.rvs(size)


class Q(Q_Base):
    """ Define general proposal as being Gaussian, centered on x_cond, 
        with variance equal to 0.1
    """
    
    def logpdf(self, x, x_cond):
        return -1/(2*0.1) * (x - x_cond)**2
        
    def rvs(self, x_cond):
        return x_cond + np.sqrt(0.1) * np.random.randn()


class L(L_Base):
    """ Define L-kernel as being Gaussian, centered on x_cond, 
        with variance equal to 0.1
    """
    
    def logpdf(self, x, x_cond):
        return -1/(2*0.1) * (x - x_cond)**2

# No. samples and iterations
N = 5000
K = 10

p = Target()
q0 = Q0()
q = Q()
l = L()

def test_sampler():
    """ Test that we can sample from a multi-modal distribution

    """

    # Run samplers
    smc_opt_gmm = SMC_OPT_GMM(N, 1, p, q0, K, q, L_components=2)
    smc_opt_gmm.generate_samples()

    assert np.allclose(smc_opt_gmm.mean_estimate_EES[-1], 0, atol=0.5)
    assert np.allclose(smc_opt_gmm.var_estimate_EES[-1], 10, atol=0.5)
