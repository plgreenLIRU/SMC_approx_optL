import numpy as np
from scipy.stats import multivariate_normal as Normal_PDF
import sys
sys.path.append('..')  # noqa
from Joint_PDF import *


"""
Testing for Joint_PDF class

P.L.Green
"""

N = 10000  # No. samples

# Create pdfs
p1 = Normal_PDF(mean=-3, cov=1)
p2 = Normal_PDF(mean=3, cov=1)
p = Joint_PDF([p1, p2])

# True samples
X1_true = p1.rvs(N)
X2_true = p2.rvs(N)

# Samples from Joint_PDF
X = p.rvs(N)


def test_samples():

    assert(np.mean(X1_true) - np.mean(X[:, 0]) < 0.1)
    assert(np.mean(X2_true) - np.mean(X[:, 1]) < 0.1)
    assert(np.var(X1_true) - np.var(X[:, 0]) < 0.1)
    assert(np.var(X2_true) - np.var(X[:, 1]) < 0.1)


def test_logp():

    logp_true = p1.logpdf(X[:, 0]) + p2.logpdf(X[:, 1])
    logp = p.logpdf(X)

    assert np.array_equal(logp_true, logp)
