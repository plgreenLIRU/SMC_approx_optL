import numpy as np
import sys
sys.path.append('SDOF_example')   # noqa
from SDOF_Class import SDOF
from SDOF_Log_Posterior_Class import SDOF_Log_Posterior as LogP
from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF

"""
Testing for SDOF Class.

P.L.Green
"""

# Define SDOF system
sdof = SDOF(m=0.5, c=0.2, k=3., h=0.1)


def test_free_vibration():
    """ Simulate free vibration response to initial displacement.

    """

    N = 1000                # No. points to simulate
    x = np.zeros(N)         # Free vibration
    sdof.S0 = np.array([1, 0])   # Initial state
    S, t = sdof.sim(x)

    # Analytical solution
    wn = np.sqrt(sdof.k / sdof.m)        # Natural freq.
    zeta = sdof.c / (2*sdof.m*wn)        # Damping ratio
    wd = wn * np.sqrt(1 - zeta**2)       # Damped natural freq.
    y_val = sdof.S0[0] * np.exp(-zeta * wn * t) * np.cos(wd * t)

    assert np.allclose(S[:, 0], y_val, atol=0.1)
