import numpy as np
from SMC_BASE import Target_Base

class SDOF_Log_Posterior(Target_Base):

    """
    Description
    -----------
    log posterior for linear single-degree-of-freedom system using
    displacement measurements.

    Parameters
    ----------
    x : force excitation

    y_obs : observed displacement time history

    q0 : List of independent priors, each of which needs a
    logpdf method.

    sdof : instance of SDOF object

    sigma : noise standard deviation

    theta : 2D array of model parameters to be looked at in a loop.
        First column is stiffness and second column is damping.

    """

    def __init__(self, x, y_obs, q0, sdof):

        self.x = x
        self.y_obs = np.vstack(y_obs)
        self.q0 = q0
        self.sdof = sdof

    def logpdf(self, theta):

        # If theta is 1D, convert to 2D
        if len(np.shape(theta)) == 1:
            theta = np.array([theta])

        # No. of parameter vectors to simulate
        N_theta = np.shape(theta)[0]

        # Initialise array of log-posterior values
        logp = np.zeros(N_theta)

        # Loop over parameter vectors
        for i in range(N_theta):

            # Stiffness, damping and noise std
            self.sdof.k = theta[i][0]
            self.sdof.c = theta[i][1]
            sigma = theta[i][2]

            # Simulate system and extract displacement
            S, t = self.sdof.sim(self.x)
            y = np.vstack(S[:, 0])

            # Find the log prior
            logq0 = self.q0.logpdf(theta[i, :])

            # Find the log-likelihood
            N = len(self.x)
            dy = y - self.y_obs
            logl = (-N / 2 * np.log(2 * np.pi * sigma**2)
                    - 1 / (2 * sigma**2) * dy.T @ dy)

            # Find the log posterior
            logp[i] = logl + logq0

        return logp
