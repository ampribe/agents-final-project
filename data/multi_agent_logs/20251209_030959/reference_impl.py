import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import logging

def solve(problem: dict) -> dict:
    """Compute Laplacian of graph given in CSR format.
    Returns a dict with key "laplacian" containing CSR components as lists.
    """
    try:
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"]
        )
        normed = problem.get("normed", False)
    except Exception as e:
        logging.error(f"Failed to reconstruct input CSR matrix: {e}")
        return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": problem.get("shape", (0, 0))}}

    try:
        L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
        if not isinstance(L, scipy.sparse.csr_matrix):
            L = L.tocsr()
        L.eliminate_zeros()
    except Exception as e:
        logging.error(f"Failed to compute Laplacian: {e}")
        return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": problem["shape"]}}

    return {
        "laplacian": {
            "data": L.data.tolist(),
            "indices": L.indices.tolist(),
            "indptr": L.indptr.tolist(),
            "shape": L.shape,
        }
    }
