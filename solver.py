import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_poisson(U: np.ndarray, 
                  rho: np.ndarray = None, 
                  eps: np.ndarray = None
                  ) -> np.ndarray:
    """
    Solve a 2D electrostatic Poisson equation on a rectangular grid
    using finite differences and sparse linear algebra.

    Equation (discrete form):
        ∇ · (ε ∇φ) = -ρ,
    with boundary conditions set by prescribed values in U.

    Parameters
    ----------
    U : (Nr, Nc) ndarray
        Potential matrix with Dirichlet boundary conditions.
        Nonzero entries on the boundary are treated as fixed.
    rho : (Nr, Nc) ndarray, optional
        Charge density distribution. Defaults to zeros.
    eps : (Nr, Nc) ndarray, optional
        Spatially varying dielectric distribution. Defaults to ones.

    Returns
    -------
    phi : (Nr, Nc) ndarray
        Solution for the potential satisfying the discrete Poisson equation.

    Notes
    -----
    - Uses a 5-point stencil with variable ε.
    - Boundaries: outer edges are fixed; additionally, any non-zero entry
      of U is treated as a Dirichlet boundary point.
    - Internally flattens the 2D arrays to build a sparse matrix of size
      (Nr*Nc) x (Nr*Nc).
    """
    Nr, Nc = U.shape

    if rho is None:
        rho = np.zeros_like(U, dtype=float)
    if eps is None:
        eps = np.ones_like(U, dtype=float)

    # pad eps to simplify neighbor indexing
    epsil = np.r_[np.ones(Nc), eps.ravel(), 1]

    # identify Dirichlet boundary nodes
    bound = np.zeros_like(U, dtype=bool)
    bound[0, :] = bound[-1, :] = True
    bound[:, 0] = bound[:, -1] = True
    bound[np.abs(U) > 1e-7] = True
    bound = bound.ravel()

    # diagonal entries: central coefficients
    main_diag = -(epsil[Nc:Nr*Nc+Nc] + epsil[:Nr*Nc] +
                  epsil[1+Nc:Nr*Nc+Nc+1] + epsil[1:Nr*Nc+1])
    east_diag = 0.5*(epsil[1:Nr*Nc] + epsil[1+Nc:Nr*Nc+Nc])
    west_diag = 0.5*(epsil[1+Nc:Nr*Nc+Nc] + epsil[1:Nr*Nc])
    north_diag = 0.5*(epsil[1+Nc:Nr*Nc+1] + epsil[Nc:Nr*Nc])
    south_diag = 0.5*(epsil[Nc:Nr*Nc] + epsil[1+Nc:Nr*Nc+1])

    # apply boundary conditions
    main_diag[bound] = 1.0
    east_diag[bound[1:Nr*Nc]] = 0.0
    west_diag[bound[:-1]] = 0.0
    north_diag[bound[Nc:Nr*Nc]] = 0.0
    south_diag[bound[:-Nc]] = 0.0

    # build sparse matrix with 5-point stencil
    A = diags(
        [main_diag, east_diag, west_diag, north_diag, south_diag],
        [0, -1, 1, -Nc, Nc],
        format="csr"
    )

    # right-hand side
    b = U.ravel() - rho.ravel()

    # solve linear system
    phi = spsolve(A, b).reshape(Nr, Nc)
    
    return phi
