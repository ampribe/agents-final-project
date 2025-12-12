import logging
import numpy as np
import scipy.sparse

def solve(problem: dict) -> dict:
    """Compute the graph Laplacian of a CSR matrix.

    Parameters
    ----------
    problem: dict
        Dictionary containing CSR components of the adjacency matrix under
        keys "data", "indices", "indptr", "shape" and a boolean flag
        "normed" indicating whether to compute the normalized Laplacian.

    Returns
    -------
    dict
        Dictionary with a single key "laplacian" whose value is another
        dictionary with CSR components of the resulting Laplacian matrix
        ("data", "indices", "indptr", "shape").  In case of failure all
        component lists are empty and "shape" mirrors the input shape (or
        defaults to ``(0, 0)``).
    """
    # Reconstruct the input sparse matrix safely
    try:
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )
        normed = bool(problem.get("normed", False))
    except Exception as e:
        logging.error(f"Failed to reconstruct input CSR matrix: {e}")
        return {
            "laplacian": {
                "data": [],
                "indices": [],
                "indptr": [],
                "shape": problem.get("shape", (0, 0)),
            }
        }

    # Compute the Laplacian using scipy's helper
    try:
        L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
        # Ensure CSR format for a canonical representation
        if not isinstance(L, scipy.sparse.csr_matrix):
            L_csr = L.tocsr()
        else:
            L_csr = L
        L_csr.eliminate_zeros()
    except Exception as e:
        logging.error(f"scipy.sparse.csgraph.laplacian failed: {e}")
        return {
            "laplacian": {
                "data": [],
                "indices": [],
                "indptr": [],
                "shape": problem["shape"],
            }
        }

    # Convert CSR components to plain Python lists to avoid ambiguous truth-value checks
    solution = {
        "laplacian": {
            "data": L_csr.data.tolist(),
            "indices": L_csr.indices.tolist(),
            "indptr": L_csr.indptr.tolist(),
            "shape": L_csr.shape,
        }
    }
    return solution
