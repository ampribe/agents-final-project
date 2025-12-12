import logging
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


def solve(problem):
    """Compute graph Laplacian from CSR representation.

    Parameters
    ----------
    problem : dict
        Dictionary containing CSR components of the adjacency matrix and a
        boolean ``normed`` flag. Expected keys:
        - "data": list or array of non‑zero values
        - "indices": list or array of column indices
        - "indptr": list or array of row pointers
        - "shape": [n, n] or tuple (n, n)
        - "normed": bool, if True compute the symmetric normalized Laplacian.

    Returns
    -------
    dict
        Dictionary with a single key ``"laplacian"`` mapping to another dict
        holding CSR components of the resulting Laplacian matrix. Keys are
        ``"data"``, ``"indices"``, ``"indptr"``, and ``"shape"``. In case of
        failure the CSR components are empty lists and ``shape`` defaults to
        the input shape or ``(0, 0)``.
    """
    # Basic validation / extraction
    try:
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        shape = tuple(problem["shape"])
        normed = bool(problem.get("normed", False))
    except Exception as e:
        logging.error(f"Problem dictionary missing required fields or invalid: {e}")
        return {
            "laplacian": {
                "data": [],
                "indices": [],
                "indptr": [],
                "shape": problem.get("shape", (0, 0)),
            }
        }

    # Reconstruct the input adjacency matrix in CSR format
    try:
        graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    except Exception as e:
        logging.error(f"Failed to build CSR matrix from input: {e}")
        return {
            "laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}
        }

    # Compute the Laplacian using SciPy
    try:
        L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
        # Ensure CSR format
        if not isinstance(L, scipy.sparse.csr_matrix):
            L = L.tocsr()
        L.eliminate_zeros()
    except Exception as e:
        logging.error(f"scipy.sparse.csgraph.laplacian failed: {e}")
        return {
            "laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}
        }

    # Return components as plain Python lists / tuple
    solution = {
        "laplacian": {
            "data": L.data.tolist(),
            "indices": L.indices.tolist(),
            "indptr": L.indptr.tolist(),
            "shape": L.shape,
        }
    }
    return solution
