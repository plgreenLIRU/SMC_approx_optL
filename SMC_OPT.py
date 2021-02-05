import numpy as np
from SMC_BASE import *
from scipy.stats import multivariate_normal as Normal_PDF
from scipy import linalg
from sklearn.decomposition import PCA

"""
A class of SMC sampler that builds on the SMC base class by allowing a
Gaussian approximation of the optimum L-kernel.

P.L.Green & A.Lee
"""


class SMC_OPT(SMC_BASE):

    def __init__(self, N, D, p, q0, K, q, PCA = None, t = None):
        """ Initialiser class method
        """
        self.PCA = PCA
        self.t = t
        # Initiate standard SMC sampler but with no L-kernel defined
        super().__init__(N, D, p, q0, K, q, L=None)

    def find_optL(self, x, x_new):
        """ Generate a Gaussian approximation of the optimum L-kernel.
        """
        
        
        if self.PCA == 'eigh':
            K1 = np.cov(x.T)
            V, Phi = linalg.eigh(K1)
            Phi = np.flip(Phi, 1)
            Phi = Phi[:, 0:self.t]
            Z1 = x @ Phi

            K2 = np.cov(x_new.T)
            U, Theta = linalg.eigh(K2)
            Theta = np.flip(Theta, 1)
            Theta = Theta[:, 0: self.t]
            Z2 = x_new @ Theta

            Z3 = np.hstack([Z1,Z2])
            K = np.cov(Z3.T)
            mu = np.mean(Z3, axis = 0)

            A = np.zeros((2 * self.D, 2 * self.t))
            A[0:self.D, 0:self.t] = Phi
            A[self.D: 2 * self.D, self.t: 2 * self.t] = Theta
            Sigma_X = A @ K @ A.T + 0.001 ** 2 * np.eye(2 * self.D)
            mu_X = A @ mu

            
        elif self.PCA == 'naive':
            K1 = np.cov(x.T)
            Q1 = np.eye(self.D)
            for i in range(500):
                Q, R = np.linalg.qr(K1)
                Q1 = Q1 @ Q
                K1 = R @ Q
            Phi = Q1[:, 0:self.t]
            Z1 = x @ Phi
            
            K2 = np.cov(x_new.T)
            Q2 = np.eye(self.D)
            for i in range(500):
                Q, R = np.linalg.qr(K2)
                Q2 = Q2 @ Q
                K2 = R @ Q
            Theta = Q1[:, 0:self.t]
            Z2 = x_new @ Theta
            
            Z3 = np.hstack([Z1,Z2])
            K = np.cov(Z3.T)
            mu = np.mean(Z3, axis = 0)
            
            A = np.zeros((2 * self.D, 2 * self.t))
            A[0:self.D, 0:self.t] = Phi
            A[self.D: 2 * self.D, self.t: 2 * self.t] = Theta
            Sigma_X = A @ K @ A.T + 0.001 ** 2 * np.eye(2 * self.D)
            mu_X = A @ mu
            
        elif self.PCA == 'SVD':
            pca = PCA(n_components = self.t, svd_solver='full')
            Z1 = pca.fit_transform(x)
            Phi = pca.components_.T
            Z2 = pca.fit_transform(x_new)
            Theta = pca.components_.T
            Z3 = np.hstack([Z1,Z2])
            K = np.cov(Z3.T)
            mu = np.mean(Z3, axis = 0)
            A = np.zeros((2 * self.D, 2 * self.t))
            A[0:self.D, 0:self.t] = Phi
            A[self.D: 2 * self.D, self.t: 2 * self.t] = Theta
            Sigma_X = A @ K @ A.T + 0.001 ** 2 * np.eye(2 * self.D)
            mu_X = A @ mu
        else:
            X = np.hstack([x, x_new])
            Sigma_X = np.cov(np.transpose(X))
            mu_X = np.mean(X, axis=0)

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
            self.Sigma = Sigma

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