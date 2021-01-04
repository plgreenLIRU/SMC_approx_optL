from SDOF_Class import SDOF
from matplotlib import pyplot as plt
import numpy as np
import dill

"""
Creates some training data for the SDOF example and saves it in a .dat file.

P.L.Green
"""

# Generate system response using random forcing
N = 100         
F = np.random.randn(N)
S0 = np.array([0, 0])
sdof = SDOF(m=1., c=0.1, k=2., h=0.1, S0=np.zeros(2))

# Create observation data by corrupting true response with noise
S, t = sdof.sim(F)
sigma = 0.05
y_true = S[:, 0]
y_obs = y_true + sigma * np.random.randn(N)

# Plot
fig, ax = plt.subplots()
ax.plot(t, y_obs, label='Observed')
ax.plot(t, y_true, label='True')
ax.legend()
ax.set_ylabel('Displacement')

# Save training data
file_name = 'training_data.dat'
file = open(file_name, 'wb')
dill.dump([F, y_true, y_obs, t, sdof, sigma], file)
file.close()

plt.show()
