import numpy as np
from scipy.stats import multivariate_normal as Normal_PDF


class GMM_PDF():

    def __init__(self, D, means, vars, weights, n_components):
        """ Initiate with mean and standard deviation.

            Note that, with this class, it is assumed that the mean and
            variance is a function of 'x_cond' in the following.

        """

        # Assign variables to object instance
        self.means = means
        self.vars = vars
        self.weights = weights
        self.n_components = n_components
        self.D = D

        # Define each components as a seperate normal pdf
        self.pdfs = []
        for c in range(n_components):
            self.pdfs.append(Normal_PDF(mean=self.means[c],
                                        cov=self.vars[c]))

    def sample(self, N=1):
        """ Generate samples.
        Currently only set up for problems with 2 components.

        """

        X = np.zeros([N, self.D])
        for i in range(N):
            u = np.random.rand()
            if u < self.weights[0]:
                X[i] = self.pdfs[0].rvs()
            else:
                X[i] = self.pdfs[1].rvs()

        return X

    def logpdf(self, x):
        """ Return the log pdf (note we've missed out the normalising
        constant).

        """

        p = 0
        for c in range(self.n_components):
            if self.D == 1:
                p += self.weights[c] * np.exp(-1 / (2 * self.vars[c]) *
                                              (x - self.means[c])**2)
            else:
                pass

        lp = np.log(p)

        return lp
