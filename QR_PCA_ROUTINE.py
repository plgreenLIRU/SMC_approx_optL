from scipy import linalg
import numpy as np

def QR_PCA(X, t, D, unique_basis = False):
    v, phi = linalg.eigh(X, subset_by_index=([2*D - t, 2*D - 1]))
    if unique_basis == True and t > 1:
        n = 0
        for i in range(2*D - 1):
            a,b = v[i], v[i+1]
            if abs(a - b) <= 10e-8:
                n += 1
            else:
                break
        phi = phi[:, n:]
    phi = np.flip(phi, 0)
    return phi
