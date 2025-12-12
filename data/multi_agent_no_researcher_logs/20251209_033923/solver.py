import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


def solve(problem):
    """Compute the Laplacian of a graph given in CSR format.

    Parameters
    ----------
    problem: dict
        Dictionary with keys:
        - "data": list of non‑zero edge weights
        - "indices": column indices for the data
        - "indptr": row pointer array
        - "shape": [n, n] representing matrix dimensions
        - "normed": bool, whether to compute the symmetric normalized Laplacian

    Returns
    -------
    dict
        Dictionary with a single key "laplacian" containing CSR components of the
        resulting Laplacian matrix (data, indices, indptr, shape).
    """
    try:
        # Reconstruct the adjacency matrix (could be weighted)
        A = sp.csr_matrix((problem["data"], problem["indices"], problem["indptr"]),
                           shape=tuple(problem["shape"]))
        normed = bool(problem.get("normed", False))

        # Compute Laplacian using scipy's trusted implementation
        L = csgraph.laplacian(A, normed=normed)
        # Ensure CSR format
        if not isinstance(L, sp.csr_matrix):
            L = L.tocsr()
        # Remove explicit zeros for a canonical representation
        L.eliminate_zeros()

        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": list(L.shape),
            }
        }
    except Exception:
        # On failure return an empty sentinel matching the expected structure
        shape = problem.get("shape", (0, 0))
        return {
            "laplacian": {
                "data": [],
                "indices": [],
                "indptr": [],
                "shape": shape,
            }
        }
