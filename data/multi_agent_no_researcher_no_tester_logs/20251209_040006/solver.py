import logging
import numpy as np
import scipy.sparse


def solve(problem):
    """Compute the graph Laplacian from a CSR representation.

    Parameters
    ----------
    problem : dict
        Dictionary containing the CSR components of the adjacency matrix ``A``
        and a boolean ``normed`` flag. Expected keys:
        - "data": list or array of non‑zero values of ``A``
        - "indices": column indices for the data values
        - "indptr": row pointer indices
        - "shape": sequence (n, n) defining matrix dimensions
        - "normed": bool indicating whether to compute the symmetric normalized
          Laplacian (True) or the combinatorial Laplacian (False).

    Returns
    -------
    dict
        Dictionary with a single key ``"laplacian"`` holding another dict with
        CSR components of the resulting Laplacian matrix:
        ``{"data": ..., "indices": ..., "indptr": ..., "shape": ...}``
        In case of any error an empty CSR structure is returned, matching the
        behaviour expected by the validator.
    """
    # ---------------------------------------------------------------------
    # Reconstruct the input adjacency matrix
    # ---------------------------------------------------------------------
    try:
        A = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=tuple(problem["shape"]),
        )
        normed = bool(problem["normed"])
    except Exception as exc:  # pragma: no cover
        logging.error("Failed to build CSR matrix from input: %s", exc)
        return {
            "laplacian": {
                "data": [],
                "indices": [],
                "indptr": [],
                "shape": tuple(problem.get("shape", (0, 0))),
            }
        }

    # ---------------------------------------------------------------------
    # Compute the Laplacian
    # ---------------------------------------------------------------------
    try:
        if normed:
            # Symmetric normalized Laplacian: I - D^{-1/2} A D^{-1/2}
            # Compute degree vector
            deg = np.asarray(A.sum(axis=1)).flatten()
            # Guard against zero degree entries
            inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
            D_inv_sqrt = scipy.sparse.diags(inv_sqrt)
            I = scipy.sparse.identity(A.shape[0], format="csr", dtype=A.dtype)
            L = I - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            # Combinatorial Laplacian: D - A
            deg = np.asarray(A.sum(axis=1)).flatten()
            D = scipy.sparse.diags(deg)
            L = D - A
        # Ensure CSR format and remove explicit zeros for a canonical form
        L = L.tocsr()
        L.eliminate_zeros()
    except Exception as exc:  # pragma: no cover
        logging.error("Failed to compute Laplacian: %s", exc)
        return {
            "laplacian": {
                "data": [],
                "indices": [],
                "indptr": [],
                "shape": tuple(problem["shape"]),
            }
        }

    # ---------------------------------------------------------------------
    # Prepare output in plain Python lists for JSON serialisation
    # ---------------------------------------------------------------------
    return {
        "laplacian": {
            "data": L.data.tolist(),
            "indices": L.indices.tolist(),
            "indptr": L.indptr.tolist(),
            "shape": L.shape,
        }
    }

# Export the function directly as the module entry point
__all__ = ["solve"]
