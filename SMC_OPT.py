import numpy as np
from SMC_BASE import *
from QR_PCA_ROUTINE import *
from scipy.stats import multivariate_normal as Normal_PDF

"""
A class of SMC sampler that builds on the SMC base class by allowing a
Gaussian approximation of the optimum L-kernel.

P.L.Green
"""


class SMC_OPT(SMC_BASE):

    def __init__(self, N, D, p, q0, K, q, PCA = [False, 0, False]):
        """ Initialiser class method

        """
        self.QR_PCA, self.t, self.unique_basis = PCA[0], PCA[1], PCA[2]

        # Initiate standard SMC sampler but with no L-kernel defined
        super().__init__(N, D, p, q0, K, q, L=None)

    def find_optL(self, x, x_new):
        """ Generate a Gaussian approximation of the optimum L-kernel.

        """

        # Collect x and x_new together into X
        X = np.hstack([x, x_new])
        self.X = X

        # Directly estimate the mean and
        # covariance matrix of X
        
        Sigma_X = np.cov(np.transpose(X))
        
        if self.QR_PCA == True:
            Phi = QR_PCA(Sigma_X, t = self.t, D = self.D, unique_basis = self.unique_basis)
            Z = np.zeros((self.N, self.t))
            for i in range(self.N):
                Z[i] = Phi.T @ X[i]
            mu_Z = np.mean(Z, axis=0)
            Sigma_Z = np.cov(np.transpose(Z))
            mu_X = Phi @ mu_Z
            Sigma_X = Phi @ Sigma_Z @ Phi.T
        else:
            mu_X = np.mean(X, axis=0)
        
        self.sig = Sigma_X

        # Find mean of joint distribution (p(x, x_new))
        mu_x, mu_xnew = mu_X[0:self.D], mu_X[self.D:2 * self.D]

        # Find covariance matrix of joint distribution (p(x, x_new))
        (Sigma_x_x,
         Sigma_x_xnew,
         Sigma_xnew_x,
         Sigma_xnew_xnew) = (Sigma_X[0:self.D, 0:self.D],
                             Sigma_X[0:self.D, self.D:2 * self.D],
                             Sigma_X[self.D:2 * self.D, 0:self.D],
                             Sigma_X[self.D:2 * self.D, self.D:2 * self.D])

        # Define new L-kernel
        def L_logpdf(x, x_cond):

            # Mean of approximately optimal L-kernel
            mu = (mu_x + Sigma_x_xnew @ np.linalg.inv(Sigma_xnew_xnew) @
                  (x_cond - mu_xnew))

            # Variance of approximately optimal L-kernel
            Sigma = (Sigma_x_x - Sigma_x_xnew @
                     np.linalg.inv(Sigma_xnew_xnew) @ Sigma_xnew_x)

            # Log det covariance matrix
            sign, logdet = np.linalg.slogdet(Sigma)
            log_det_Sigma = sign * logdet

            # Inverse covariance matrix
            inv_Sigma = np.linalg.inv(Sigma)

            # Find log pdf
            logpdf = (-0.5 * log_det_Sigma -
                      0.5 * (x - mu).T @ inv_Sigma @ (x - mu))

            return logpdf

        self.L = L_Kernel()
        self.L.logpdf = L_logpdf

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
                           self.L.logpdf(x=x[i], x_cond=x_new[i]) -
                           self.q.logpdf(x=x_new[i], x_cond=x[i]))

        return logw_new
