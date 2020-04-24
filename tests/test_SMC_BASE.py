import numpy as np
import sys
sys.path.append('..')  # noqa
from SMC_BASE import *
from scipy.stats import multivariate_normal as Normal_PDF
from Normal_PDF_Cond import *

"""
Testing for SMC_BASE

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


def test_resample():
    """ Initial proposal is very close to target so, after resampling, we
    should be left with a pretty representitive set of samples (whose mean
    and variance is close to the mean and variance of the target).

    """

    # A quick test, so we repeat it 100 times
    for n in range(100):
        X = q0.rvs(N)  # Generate samples
        p_logpdf_x = p.logpdf(X)  # log pdf of target
        logw = p_logpdf_x - q0.logpdf(X)  # log weights
        wn = np.vstack(smc.normalise_weights(logw))  # normalised log weights
        X_rs = smc.resample(X, p_logpdf_x, wn)[0]  # resample

        assert np.allclose(np.mean(X_rs), p.mean, atol=0.3)
        assert np.allclose(np.var(X_rs, 0), np.diag(p.cov), atol=0.3)
