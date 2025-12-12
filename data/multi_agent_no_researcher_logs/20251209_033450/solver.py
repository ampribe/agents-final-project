import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import logging

def solve(problem):
    """Solve all-pairs shortest paths for an undirected weighted graph given in CSR format.
    Returns a dict with key 'distance_matrix' as list of lists, using None for unreachable pairs.
    Handles edge cases of empty graph (n=0) and single-node graph (n=1).
    """
    # Validate required keys
    required_keys = {"data", "indices", "indptr", "shape"}
    if not required_keys.issubset(problem):
        logging.error("Problem dictionary missing required CSR components.")
        return {"distance_matrix": []}
    try:
        n = int(problem["shape"][0])
        # Build CSR matrix
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"]
        )
    except Exception as e:
        logging.error(f"Failed to construct CSR matrix: {e}")
        return {"distance_matrix": []}

    # Edge cases
    if n == 0:
        return {"distance_matrix": []}
    if n == 1:
        # Single node, distance to itself is 0.0 regardless of edges list.
        return {"distance_matrix": [[0.0]]}

    try:
        # Compute all-pairs shortest paths; graph is undirected
        dist_matrix = scipy.sparse.csgraph.shortest_path(
            csgraph=graph_csr, method="auto", directed=False, unweighted=False
        )
    except Exception as e:
        logging.error(f"scipy shortest_path failed: {e}")
        return {"distance_matrix": []}

    # Convert to list of lists with None for infinities
    distance_list = []
    for row in dist_matrix:
        distance_row = []
        for d in row:
            if np.isinf(d):
                distance_row.append(None)
            else:
                # Ensure Python float
                distance_row.append(float(d))
        distance_list.append(distance_row)

    return {"distance_matrix": distance_list}
