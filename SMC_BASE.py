import numpy as np
from abc import abstractmethod, ABC

class Target_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the target distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the target distribution, evaluated at x.

        """
        pass

class Q0_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the initial proposal distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the initial proposal, evaluated at x.
        """
        pass

    @abstractmethod
    def rvs(self, size):
        """
        Description
        -----------
        Returns samples from the initial proposal.

        Parameters
        ----------
        size : size of the sample being returned
        """
        pass

class Q_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the general proposal distribution.

    """

    @abstractmethod
    def logpdf(self, x, x_cond):
        """
        Description
        -----------
        Returns log q(x | x_cond)
        """
        pass

    @abstractmethod
    def rvs(self, x_cond):
        """
        Description
        -----------
        Returns a single sample from the proposal, q(x | x_cond).
        """

        pass

class L_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the L-kernel.

    """

    @abstractmethod
    def logpdf(self, x, x_cond):
        """
        Description
        -----------
        Returns log L(x | x_cond)
        """
        pass

class SMC():

    """
    Description
    -----------
    A base class for an SMC sampler.

    Parameters
    ----------
    N : no. of samples generated at each iteration

    D : dimension of the target distribution

    p : target distribution instance

    q0 : initial proposal instance

    K : no. iterations to run

    q : general proposal distribution instance

    L : L-kernel instance

    sampling : 'batch' or 'single_step' approach (where single_step should
        be better for high dimensional problems).

    Methods
    -------
    normalise_weights : normalises importance sampling weights

    resample : resamples from normalised importance weights

    estimate : realise importance sampling estimates of mean and
        covariance matrix of the target.

    generate_samples : runs the SMC sampler to generate weighted
        samples from the target.

    propose_sample : proposes new samples, could probably remove in the
        future.

    update_weights : updates the log weight associated with each sample
        i.e. evaluates the incremental weights.

    Author
    ------
    P.L.Green
    """

    def __init__(self, N, D, p, q0, K, q, L, sampling='batch'):

        # Assign variables to self
        self.N = N
        self.D = D
        self.p = p
        self.q0 = q0
        self.K = K
        self.q = q
        self.L = L
        self.sampling = sampling

    def normalise_weights(self, logw):
        """
        Description
        -----------
        Normalise importance weights. Note that we remove the mean here
            just to avoid numerical errors in evaluating the exponential.
            We have to be careful with -inf values in the log weights
            sometimes. This can happen if we are sampling from a pdf with
            zero probability regions, for example.

        Parameters
        ----------
        logw : array of logged importance weights

        Returns
        -------
        wn : array of normalised weights

        """

        # Identify elements of logw that are not -inf
        indices = np.invert(np.isneginf(logw))

        # Produce copy of logw, as we don't want the current method to alter
        # logw itself
        logw_copy = np.copy(logw)

        # Apply normalisation only to those elements of log that are not -inf
        logw_copy[indices] = logw_copy[indices] - np.max(logw_copy[indices])

        # Find standard weights
        w = np.exp(logw_copy)
        
        # Find normalised weights
        wn = w / np.sum(w)

        return wn

    def resample(self, x, p_logpdf_x, wn):
        """
        Description
        -----------
        Resample given normalised weights.

        Parameters
        ----------
        x : array of current samples

        p_logpdf_x : array of current target evaluations.

        wn : array or normalised weights

        Returns
        -------
        x_new : resampled values of x

        p_logpdf_x_new : log pdfs associated with x_new

        wn_new : normalised weights associated with x_new

        """

        i = np.linspace(0, self.N-1, self.N, dtype=int)  # Sample positions
        i_new = np.random.choice(i, self.N, p=wn[:, 0])   # i is resampled
        wn_new = np.ones(self.N) / self.N           # wn is reset

        # New samples
        x_new = x[i_new]
        p_logpdf_x_new = p_logpdf_x[i_new]

        return x_new, p_logpdf_x_new, wn_new

    def estimate(self, x, wn):
        """
        Description
        -----------
        Estimate some quantities of interest (just mean and covariance
            matrix for now).

        Parameters
        ----------
        x : samples from the target

        wn : normalised weights associated with the target

        Returns
        -------
        m : estimated mean

        v : estimated covariance matrix

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

        """
        Description
        -----------
        Run SMC sampler to generate weighted samples from the target.

        """

        # Initialise arrays
        x_new = np.zeros([self.N, self.D])
        lr = np.array([])

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
            lr = np.append(lr, np.sum(wn)**2 / np.sum(wn**2))
            lmbda = np.array([])
            for k_dash in range(self.k + 1):
                lmbda = np.append(lmbda, lr[k_dash] / np.sum(lr))
                self.mean_estimate_EES[self.k] += (lmbda[k_dash] *
                                                   self.mean_estimate[k_dash])
                self.var_estimate_EES[self.k] += (lmbda[k_dash] *
                                                  self.var_estimate[k_dash])

            # Record effective sample size at kth iteration
            self.Neff[self.k] = 1 / np.sum(np.square(wn))

            # Generate new samples
            if self.sampling == 'batch':

                # Resample if effective sample size is below threshold
                if self.Neff[self.k] < self.N/2:

                    self.resampling_points = np.append(self.resampling_points,
                                                       self.k)
                    x, p_logpdf_x, wn = self.resample(x, p_logpdf_x, wn)
                    logw = np.log(wn)

                # If we are using a batched sampling approach, we
                # propose new samples, across all dimensions
                for i in range(self.N):
                    x_new[i] = self.q.rvs(x_cond=x[i])

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
                logw = np.copy(logw_new)
                p_logpdf_x = np.copy(p_logpdf_x_new)

            if self.sampling == 'single_step':

                # Loop to update one dimension at a time
                for d in range(self.D):

                    x_new = np.copy(x)
                    for i in range(self.N):
                        x_new[i, d] = self.q.rvs(x[i, d])

                    # Make sure evaluations of likelihood are vectorised
                    p_logpdf_x_new = self.p.logpdf(x_new)

                    # Update log weights
                    logw_new = self.update_weights(x, x_new, logw,
                                                   p_logpdf_x,
                                                   p_logpdf_x_new, d)

                    # Find normalised weights
                    wn = self.normalise_weights(logw_new)

                    # Resample if effective sample size is below threshold
                    Neff = 1 / np.sum(np.square(wn))
                    if Neff < self.N/2:
                        [x_new,
                         p_logpdf_x_new,
                         wn] = self.resample(x_new, p_logpdf_x_new, wn)
                        logw_new = np.log(wn)

                    # Update samples, log weights, and posterior evaluations
                    x = np.copy(x_new)
                    logw = np.copy(logw_new)
                    p_logpdf_x = np.copy(p_logpdf_x_new)

        # Final quantities to be returned
        self.x = x
        self.logw = logw

    def update_weights(self, x, x_new, logw, p_logpdf_x,
                       p_logpdf_x_new, d=None):
        """
        Description
        -----------
        Used to update the log weights of a new set of samples, using the
            weights of the samples from the previous iteration.

        Parameters
        ----------
        x : samples from the previous iteration

        x_new : samples from the current iteration

        logw : low importance weights associated with x

        p_logpdf_x : log target evaluations associated with x

        p_logpdf_x_new : log target evaluations associated with x_new

        d : current dimension we are updating (only needed if single_step
            sampling is being used).

        Returns
        -------
        logw_new : log weights associated with x_new

        """

        # Initialise
        logw_new = np.vstack(np.zeros(self.N))

        # Find new weights
        if self.sampling == 'batch':
            for i in range(self.N):
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i] +
                               self.L.logpdf(x[i], x_new[i]) -
                               self.q.logpdf(x_new[i], x[i]))
        if self.sampling == 'single_step':
            for i in range(self.N):
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i] +
                               self.L.logpdf(x[i, d], x_new[i, d]) -
                               self.q.logpdf(x_new[i, d], x[i, d]))

        return logw_new
