import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

def solve(problem: dict) -> dict:
    """Reference implementation using scipy's shortest_path.
    Returns a dict with key 'distance_matrix' where unreachable distances are None.
    """
    try:
        graph = scipy.sparse.csr_matrix((problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"])
    except Exception:
        return {"distance_matrix": []}
    try:
        dist = scipy.sparse.csgraph.shortest_path(csgraph=graph, method='auto', directed=False)
    except Exception:
        return {"distance_matrix": []}
    # Convert np.inf to None for JSON friendliness
    dist_list = [[None if np.isinf(d) else float(d) for d in row] for row in dist]
    return {"distance_matrix": dist_list}
