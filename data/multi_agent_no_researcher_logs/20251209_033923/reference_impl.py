import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

def solve(problem):
    """Reference implementation for graph Laplacian computation.
    Returns a dict with key 'laplacian' containing CSR components.
    """
    try:
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"]
        )
        normed = problem["normed"]
    except Exception as e:
        # Return empty components on failure
        return {
            "laplacian": {
                "data": [],
                "indices": [],
                "indptr": [],
                "shape": problem.get("shape", (0, 0)),
            }
        }
    try:
        L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
        if not isinstance(L, scipy.sparse.csr_matrix):
            L = L.tocsr()
        L.eliminate_zeros()
    except Exception as e:
        return {
            "laplacian": {"data": [], "indices": [], "indptr": [], "shape": problem["shape"]}
        }
    return {
        "laplacian": {
            "data": L.data.tolist(),
            "indices": L.indices.tolist(),
            "indptr": L.indptr.tolist(),
            "shape": L.shape,
        }
    }
