import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags


def solve_poisson(U, Rho=None, Eps=None):

    Nr, Nc = U.shape

    if Rho is None:
        Rho = np.zeros_like(U)

    if Eps is None:
        Eps = np.ones_like(U)

    Epsil = np.r_[np.ones(Nc), Eps.flatten(), 1]

    Bound = np.zeros_like(U, dtype=bool)
    Bound[0, :] = True
    Bound[Nr-1, :] = True
    Bound[:, 0] = True
    Bound[:, Nc-1] = True
    Bound[abs(U) > 1e-7] = True
    Bound = Bound.flatten()

    s = [-(Epsil[Nc:Nr*Nc+Nc] + Epsil[:Nr*Nc] +
           Epsil[1+Nc:Nr*Nc+Nc+1] + Epsil[1:Nr*Nc+1]),
         0.5*(Epsil[1:Nr*Nc] + Epsil[1+Nc:Nr*Nc+Nc]),
         0.5*(Epsil[1+Nc:Nr*Nc+Nc] + Epsil[1:Nr*Nc]),
         0.5*(Epsil[1+Nc:Nr*Nc+1] + Epsil[Nc:Nr*Nc]),
         0.5*(Epsil[Nc:Nr*Nc] + Epsil[1+Nc:Nr*Nc+1])]

    s[0][Bound] = 1
    s[1][Bound[1:Nr*Nc]] = 0
    s[2][Bound[:Nr*Nc-1]] = 0
    s[3][Bound[Nc:Nr*Nc]] = 0
    s[4][Bound[:Nr*Nc-Nc]] = 0

    S = diags(s, [0, -1, 1, -Nc, Nc])
    S = csr_matrix(S)

    b = U.flatten() - Rho.flatten()

    phi = spsolve(S, b)
    phi = phi.reshape(Nr, Nc)

    return phi


def solve_poisson_(U, Rho=None, Eps=None):

    N, M = U.shape
    assert N == M

    if Rho is None:
        Rho = np.zeros_like(U)

    if Eps is None:
        Eps = np.ones_like(U)

    Epsil = np.r_[np.ones(N), Eps.flatten(), 1]

    Bound = np.zeros_like(U, dtype=bool)
    Bound[0, :] = True
    Bound[N-1, :] = True
    Bound[:, 0] = True
    Bound[:, N-1] = True
    Bound[abs(U) > 1e-7] = True
    Bound = Bound.flatten()

    s = [-(Epsil[N:N*N+N] + Epsil[:N*N] +
           Epsil[1+N:N*N+N+1] + Epsil[1:N*N+1]),
         0.5*(Epsil[1:N*N] + Epsil[1+N:N*N+N]),
         0.5*(Epsil[1+N:N*N+N] + Epsil[1:N*N]),
         0.5*(Epsil[1+N:N*N+1] + Epsil[N:N*N]),
         0.5*(Epsil[N:N*N] + Epsil[1+N:N*N+1])]

    s[0][Bound] = 1
    s[1][Bound[1:N*N]] = 0
    s[2][Bound[:N*N-1]] = 0
    s[3][Bound[N:N*N]] = 0
    s[4][Bound[:N*N-N]] = 0

    S = diags(s, [0, -1, 1, -N, N])
    S = csr_matrix(S)

    b = U.flatten() - Rho.flatten()

    phi = spsolve(S, b)
    phi = phi.reshape(N, N)

    return phi
