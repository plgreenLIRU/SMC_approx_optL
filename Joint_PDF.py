import numpy as np


class Joint_PDF():

    def __init__(self, pdfs):
        """ Initiated with a list of PDFs.

            Note - throughout it is assumed that each pdf is 1D.

        """

        # Assign variables to object instance
        self.pdfs = pdfs
        self.N_pdfs = len(pdfs)

    def rvs(self, size=1):
        """ Generate samples.

        """

        # Loop over pdfs to produce samples
        X = np.zeros([size, self.N_pdfs])
        for i in range(self.N_pdfs):
            X[:, i] = self.pdfs[i].rvs(size)

        return X

    def logpdf(self, x, single_sample=False):
        """ Find log pdf of the joint pdf. Again, is assumed that each
            pdf is 1D.

        """

        if single_sample is True:
            lpdf = 0.0
            for i in range(self.N_pdfs):
                lpdf += self.pdfs[i].logpdf(x[i])
        else:
            N = np.shape(x)[0]
            lpdf = np.zeros(N)
            for n in range(N):
                for i in range(self.N_pdfs):
                    lpdf[n] += self.pdfs[i].logpdf(x[n][i])

        return lpdf
