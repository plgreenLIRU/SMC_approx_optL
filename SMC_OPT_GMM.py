import numpy as np
from SMC_BASE import SMC, L_Base
from sklearn.mixture import GaussianMixture as GMM_sk
from GMM_Conditional import *

"""
A class of SMC sampler that builds on the SMC base class by allowing a
Gaussian mixture approximation of the optimum L-kernel.

P.L.Green
"""


class SMC_OPT_GMM(SMC):

    def __init__(self, N, D, p, q0, K, q, L_components):
        """ Initialiser class method

        """

        # No. components in the Gaussian Mixture Model
        self.L_components = L_components

        # Initiate standard SMC sampler but with no L-kernel defined
        super().__init__(N, D, p, q0, K, q, L=None)

    def find_optL(self, x, x_new):
        """ Generate a Gaussian mixture approximation of the optimum L-kernel.

        """

        # Collect x and x_new together into X
        X = np.hstack([x, x_new])

        # Fit Gaussian Mixture Model
        gmm = GMM_sk(n_components=self.L_components)
        gmm.fit(X)

        # Find conditional distributions of GMM
        gmm_cond = GMM_Conditional(means=gmm.means_,
                                   covariances=gmm.covariances_,
                                   weights=gmm.weights_,
                                   n_components=self.L_components,
                                   D1=self.D, D2=self.D)

        # Assign conditional GMM to L-kernel
        self.L = gmm_cond

    def update_weights(self, x, x_new, logw, p_log_pdf_x, p_log_pdf_x_new):
        """ Overwrites the method in the base-class

        """

        # Initialise arrays
        logw_new = np.vstack(np.zeros(self.N))

        # Find approximation of the optimum L-kernel
        self.find_optL(x, x_new)

        # Find new weights
        for i in range(self.N):

            logw_new[i] = (logw[i] +
                           p_log_pdf_x_new[i] -
                           p_log_pdf_x[i] +
                           self.L.logpdf(x[i], x_new[i]) -
                           self.q.logpdf(x_new[i], x[i]))

        return logw_new
