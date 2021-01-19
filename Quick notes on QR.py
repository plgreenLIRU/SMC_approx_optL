
"""
Some quick notes on the QR algorithm, and how it can be used in practice
"""
import numpy as np
from scipy import linalg
np.random.seed(1)
A = 10 * np.random.rand(5,5)
A = A.T @ A
A_ = np.copy(A)
Q1 = np.eye(5)

for i in range(20):
    '''Simple loop for 20 iterations of algorithm without shifts'''
    Q, R = linalg.qr(A)
    A = R @ Q
    Q1 = Q1 @ Q #lim_{i \to \infty} Q_i = I, so this product can be done cheaply once we recognise convergence :)

print('Eigensolution via loop:' + '\n')
print(np.diag(A))
print(Q1)
test = linalg.eig(A_)
print()
print('Eigensolution via scipy:' + '\n')
for i in range(2):
    print(test[i])
H = linalg.hessenberg(A_)
for i in range(30):
    Q, R = linalg.qr(H)
    H = R @ Q
print()
print('Eigenvals of Hessenberg matrix:')
print(np.diag(H))
''' The eigenvectors of H are not the same as A, obviously.

However I BELIEVE you can extract the eigenvectors of A from H using reflections
but I'm not so sure about this.

Upper Hessenberg of SPSD is also tri-diagonal so that's a LOT of zeros.'''