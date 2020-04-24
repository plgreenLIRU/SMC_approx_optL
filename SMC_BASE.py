import numpy as np
import matplotlib.pyplot as plt
import copy

"""
A base class for an SMC sampler.

P.L.Green
"""


class SMC_BASE():

    def __init__(self, N, D, p, q0, K, q, L):

        """ Initialiser class method

        """

        # Assign variables to self
        self.N = N
        self.D = D
        self.p = p
        self.q0 = q0
        self.K = K
        self.q = q
        self.L = L

    def normalise_weights(self, logw):
        """ Normalise importance weights. Note that we remove the mean here
            just to avoid numerical errors in evaluating the exponential.

            We have to be careful with -inf values in the log weights
            sometimes. This can happen if we are sampling from a pdf with
            zero probability regions, for example.

        """

        # Identify elements of logw that are not -inf
        indices = np.invert(np.isneginf(logw))

        # Apply normalisation only to those elements of log that are not -inf
        logw[indices] = logw[indices] - np.max(logw[indices])

        # Find standard weights
        w = np.exp(logw)

        return w / np.sum(w)

    def resample(self, x, p_logpdf_x, wn):
        """ Resample given normalised weights.

            p_logpdf is an array of current target evaluations.
        """

        i = np.linspace(0, self.N-1, self.N, dtype=int)  # Sample positions
        i_new = np.random.choice(i, self.N, p=wn[:, 0])   # i is resampled
        wn_new = np.ones(self.N) / self.N           # wn is reset

        return x[i_new], p_logpdf_x[i_new], wn_new

    def estimate(self, x, wn):
        """ Estimate some quantities of interest (just mean and
            covariance matrix for now).

        """

        # Estimate the mean
        m = wn.T @ x

        # Remove the mean from our samples then estimate the variance
        x = x - m

        if self.D == 1:
            v = wn.T @ np.square(x)
        else:
            v = np.zeros([self.D, self.D])
            for i in range(self.N):
                xv = x[i][np.newaxis]  # Make each x into a 2D array
                v += wn[i] * xv.T @ xv

        return m, v

    def generate_samples(self):
        """ Run SMC sampler to generate weighted samples from the target.

        """

        # Initialise arrays
        x_new = np.zeros([self.N, self.D])
        l = np.array([])

        # Initilise estimates of target mean and covariance matrix
        self.mean_estimate = np.zeros([self.K, self.D])
        self.mean_estimate_EES = np.zeros([self.K, self.D])
        if self.D == 1:
            self.var_estimate = np.zeros([self.K, self.D])
            self.var_estimate_EES = np.zeros([self.K, self.D])
        else:
            self.var_estimate = np.zeros([self.K, self.D, self.D])
            self.var_estimate_EES = np.zeros([self.K, self.D, self.D])

        # Used to record the effective sample size and the points
        # where resampling occurred.
        self.Neff = np.zeros(self.K)
        self.resampling_points = np.array([])

        # Sample from prior and find initial evaluations of the
        # target and the prior. Note that, be default, we keep
        # the log weights vertically stacked.
        x = np.vstack(self.q0.rvs(size=self.N))
        p_logpdf_x = np.vstack(self.p.logpdf(x))
        p_q0_x = np.vstack(self.q0.logpdf(x))

        # Find weights of prior samples
        logw = p_logpdf_x - p_q0_x

        # Main sampling loop
        for self.k in range(self.K):

            print('\nIteration :', self.k)

            # Find normalised weights and realise estimates
            wn = self.normalise_weights(logw)
            (self.mean_estimate[self.k],
             self.var_estimate[self.k]) = self.estimate(x, wn)

            # EES recycling scheme
            l = np.append(l, np.sum(wn)**2 / np.sum(wn**2))
            lmbda = np.array([])
            for k_dash in range(self.k + 1):
                lmbda = np.append(lmbda, l[k_dash] / np.sum(l))
                self.mean_estimate_EES[self.k] += (lmbda[k_dash] *
                                                   self.mean_estimate[k_dash])
                self.var_estimate_EES[self.k] += (lmbda[k_dash] *
                                                  self.var_estimate[k_dash])

            # Resample if effective sample size is below threshold
            self.Neff[self.k] = 1 / np.sum(np.square(wn))
            if self.Neff[self.k] < self.N/2:

                self.resampling_points = np.append(self.resampling_points,
                                                   self.k)
                x, p_logpdf_x, wn = self.resample(x, p_logpdf_x, wn)
                logw = np.log(wn)

            # Generate new samples
            for i in range(self.N):
                x_new[i] = self.propose_sample(x_cond=x[i])

            # Make sure evaluations of likelihood are vectorised
            p_logpdf_x_new = self.p.logpdf(x_new)

            # Update log weights
            logw_new = self.update_weights(x, x_new, logw, p_logpdf_x,
                                           p_logpdf_x_new)

            # Make sure that, if p.logpdf(x_new) is -inf, then logw_new
            # will also be -inf. Otherwise it is returned as NaN.
            for i in range(self.N):
                if p_logpdf_x_new[i] == -np.inf:
                    logw_new[i] = -np.inf
                elif logw[i] == -np.inf:
                    logw_new[i] = -np.inf

            # Update samples, log weights, and posterior evaluations
            x = np.copy(x_new)
            logw = copy.deepcopy(logw_new)
            p_logpdf_x = copy.deepcopy(p_logpdf_x_new)

        # Final quantities to be returned
        self.x = x
        self.logw = logw

    def propose_sample(self, x_cond):
        """ Method used specifically to propose new samples.

        """

        # New sample
        x_new = self.q.rvs(x_cond=x_cond)

        return x_new

    def update_weights(self, x, x_new, logw, p_logpdf_x, p_logpdf_x_new):
        """ Used to update the log weights.

        """

        # Initialise
        logw_new = np.vstack(np.zeros(self.N))

        # Find new weights
        for i in range(self.N):
            logw_new[i] = (logw[i] +
                           p_logpdf_x_new[i] -
                           p_logpdf_x[i] +
                           self.L.logpdf(x=x[i], x_cond=x_new[i]) -
                           self.q.logpdf(x=x_new[i], x_cond=x[i]))

        return logw_new
