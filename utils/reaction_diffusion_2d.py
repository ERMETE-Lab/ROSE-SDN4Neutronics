"""2D coupled reaction-diffusion solver on the unit square (finite differences + scipy.sparse)."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import qmc


def sample_parameters(
    n_samples: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((50.0, 100.0), (50.0, 100.0)),
    seed: int = 42,
) -> np.ndarray:
    """Latin-hypercube samples of (mu1, mu2), shape (n_samples, 2)."""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n_samples)
    lower = np.array([bounds[0][0], bounds[1][0]])
    upper = np.array([bounds[0][1], bounds[1][1]])
    return qmc.scale(unit, lower, upper)


def _build_laplacian(n: int, h: float) -> sparse.csr_matrix:
    """Negative Laplacian (-Δ) with homogeneous Dirichlet BC on a uniform grid."""
    n_nodes = n * n
    rows, cols, data = [], [], []

    def flat(i: int, j: int) -> int:
        return j * n + i

    inv_h2 = 1.0 / h**2
    for j in range(n):
        for i in range(n):
            k = flat(i, j)
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                rows.append(k)
                cols.append(k)
                data.append(1.0)
            else:
                rows.extend([k, k, k, k, k])
                cols.extend([k, flat(i + 1, j), flat(i - 1, j), flat(i, j + 1), flat(i, j - 1)])
                data.extend([4 * inv_h2, -inv_h2, -inv_h2, -inv_h2, -inv_h2])

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def _source_terms(
    mu1: float,
    mu2: float,
    hf: bool,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = np.meshgrid(xs, ys, indexing="ij")
    f_u = np.exp(-((x0 - 0.6) ** 2 + (y0 - 0.2 - mu2 / 200) ** 2) / 0.01)
    if hf:
        radial = np.sqrt((x0 - 0.5 - 0.02 * (mu1 - 75) / 25) ** 2 + (y0 - 0.3 - 0.02 * (mu2 - 75) / 25) ** 2)
        f_u = f_u + 2 * np.exp(-radial / 0.05)
    f_v = np.zeros_like(f_u)
    return f_u.ravel(), f_v.ravel()


def solve(
    mu1: float = 75.0,
    mu2: float = 75.0,
    hf: bool = True,
    Nh: int = 40,
    max_iter: int = 20,
    tol: float = 1e-6,
    mu: float | None = None,
):
    """
    Solve the coupled reaction-diffusion system from ExampleMF2D using finite differences.

    Parameters
    ----------
    mu1 : float
        Nonlinear coupling strength in the reaction term.
    mu2 : float
        Vertical shift of the LF Gaussian source.
    mu : float, optional
        Legacy single-parameter alias (sets mu1 = mu2 = mu).

    Returns
    -------
    solution : ndarray
        DOFs interleaved as [u0, v0, u1, v1, ...].
    coords : ndarray
        Vertex coordinates with shape (n_nodes, 3).
    """
    if mu is not None:
        mu1 = mu2 = float(mu)

    n = Nh + 1
    h = 1.0 / Nh
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([x_grid.ravel(), y_grid.ravel(), np.zeros(n * n)])

    laplacian = _build_laplacian(n, h)
    f_u, f_v = _source_terms(mu1, mu2, hf, xs, ys)

    boundary = []
    for j in range(n):
        for i in range(n):
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                boundary.append(j * n + i)
    for k in boundary:
        f_u[k] = 0.0
        f_v[k] = 0.0

    u_old = np.zeros(n * n)
    v_old = np.zeros(n * n)

    for _ in range(max_iter):
        r2 = u_old**2 + v_old**2
        reaction = 1.0 - r2
        coupling = mu1 * (r2 + 1.0)

        system = sparse.bmat(
            [
                [laplacian + sparse.diags(reaction), sparse.diags(coupling)],
                [-sparse.diags(coupling), laplacian + sparse.diags(reaction)],
            ],
            format="csr",
        )
        rhs = np.concatenate([f_u, f_v])
        stacked = spsolve(system, rhs)

        u_new = stacked[: n * n]
        v_new = stacked[n * n :]
        error = np.linalg.norm(stacked - np.concatenate([u_old, v_old]))
        u_old, v_old = u_new, v_new
        if error < tol:
            break

    solution = np.empty(2 * n * n)
    solution[0::2] = u_old
    solution[1::2] = v_old
    return solution, coords
