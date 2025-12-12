import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

def solve(problem):
    '''Reference implementation using scipy\'s shortest_path.
    Returns distance_matrix as list of lists with None for infinite distances.
    '''
    data = problem.get("data", [])
    indices = problem.get("indices", [])
    indptr = problem.get("indptr", [])
    shape = problem.get("shape", [0, 0])
    try:
        csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    except Exception:
        return {"distance_matrix": []}
    try:
        dist = scipy.sparse.csgraph.shortest_path(csgraph=csr, method='auto', directed=False)
    except Exception:
        return {"distance_matrix": []}
    dist_list = [[None if np.isinf(val) else float(val) for val in row] for row in dist]
    return {"distance_matrix": dist_list}
