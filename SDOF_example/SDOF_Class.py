import numpy as np
from matplotlib import pyplot as plt


class SDOF():

    """
    Description
    -----------
    Class for linear single-degree-of-freedom system.

    Parameters
    ----------
    m : mass
    c : damping coefficient
    k : stiffness coefficient
    h : timestep
    S0 : initial system state
    S : system state
    x : force excitation
    N : no. points to simulate

    P.L.Green
    """

    def __init__(self, m=None, c=None, k=None, h=None, S0=None):

        self.m = m
        self.c = c
        self.k = k
        self.h = h
        self.S0 = S0

    def RK4(self, S, x):
        """ rth order Runge-Kutta
        """

        k1 = self.dS(S, x)
        k2 = self.dS(S + self.h * k1 / 2, x)
        k3 = self.dS(S + self.h * k2 / 2, x)
        k4 = self.dS(S + self.h * k3, x)

        S_new = S + self.h / 6 * (k1 +
                                  2 * k2 +
                                  2 * k3 +
                                  k4)
        return S_new

    def dS(self, S, x):
        """ SDOF state-space model

        """

        dS = np.zeros(2)
        dS[0] = S[1]
        dS[1] = (1 / self.m * (x -
                               self.c*S[1] -
                               self.k*S[0]))

        return dS

    def sim(self, x):
        """ Simulate system.

        """
        N = len(x)
        S = np.zeros([N, 2])
        S[0] = self.S0
        t = np.arange(0, N) * self.h
        for i in range(N-1):
            S[i+1, :] = self.RK4(S[i, :], x[i])

        return S, t
