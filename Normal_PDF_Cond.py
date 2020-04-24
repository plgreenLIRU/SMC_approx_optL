import numpy as np


class Normal_PDF_Cond():

    def __init__(self, D, mean, cov):
        """ Initiate with mean and standard deviation.

            Note that, with this class, it is assumed that the mean and
            variance is a function of 'x_cond' in the following.

        """

        self.mean = mean
        self.cov = cov
        self.D = D

    def rvs(self, x_cond, N=1):
        """ Generate samples.

        """

        if self.D == 1:
            X = (self.mean(x_cond) +
                 np.sqrt(self.cov(x_cond)) * np.random.randn(N))
        else:
            X = np.random.multivariate_normal(self.mean(x_cond),
                                              self.cov(x_cond), N)

        return X

    def logpdf(self, x, x_cond):
        """ Return the log pdf (note we've missed out the normalising
        constant).

        """

        if self.D == 1:
            lp = -1 / (2 * self.cov(x_cond)) * (x - self.mean(x_cond))**2
        else:
            lp = -1 / 2 * ((x - self.mean(x_cond)) @
                           np.linalg.inv(self.cov(x_cond)) @
                           (x - self.mean(x_cond)).T)

        return lp

    def pdf(self, x, x_cond):
        """ Return the pdf.

        """

        if self.D == 1:
            p = (1 / (np.sqrt(2 * np.pi * self.cov(x_cond))) *
                 np.exp(-1 / (2 * self.cov(x_cond)) * (x -
                                                       self.mean(x_cond))**2))
        else:
            p = (1 / np.sqrt((2 * np.pi)**self.D *
                 np.linalg.det(self.cov(x_cond))) *
                 np.exp(-0.5 * (x - self.mean(x_cond)) @
                        np.linalg.inv(self.cov(x_cond)) @
                        (x - self.mean(x_cond)).T))

        return p
