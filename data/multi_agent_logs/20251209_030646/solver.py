import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import logging


def solve(problem: dict) -> dict:
    """Solve the all-pairs shortest path problem for an undirected weighted graph.

    Parameters
    ----------
    problem : dict
        Dictionary with CSR representation of the graph:
        - "data": list of edge weights
        - "indices": list of column indices for each weight
        - "indptr": index pointer array
        - "shape": [n, n] where n is the number of vertices

    Returns
    -------
    dict
        Dictionary with a single key ``"distance_matrix"`` whose value is a
        list‑of‑lists representing the shortest‑path distances. ``None`` denotes
        that no path exists between a pair of vertices.
    """
    # Required CSR keys
    required_keys = {"data", "indices", "indptr", "shape"}
    if not required_keys.issubset(problem):
        logging.error("Problem dictionary missing required CSR components.")
        return {"distance_matrix": []}

    try:
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        shape = problem["shape"]
        n = int(shape[0])
        # Build CSR matrix
        csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    except Exception as e:
        logging.error(f"Failed to construct CSR matrix: {e}")
        return {"distance_matrix": []}

    # Empty graph handling – return empty distance matrix
    if n == 0:
        return {"distance_matrix": []}

    try:
        # Compute all‑pairs shortest paths for an undirected graph
        dist_matrix = scipy.sparse.csgraph.dijkstra(
            csgraph=csr, directed=False, return_predecessors=False
        )
    except Exception as e:
        logging.error(f"scipy.dijkstra failed: {e}")
        return {"distance_matrix": []}

    # Convert infinities to ``None`` for JSON‑serialisation friendliness
    distance_matrix = []
    for row in dist_matrix:
        distance_row = []
        for val in row:
            if np.isinf(val):
                distance_row.append(None)
            else:
                # Ensure plain Python float, not numpy scalar
                distance_row.append(float(val))
        distance_matrix.append(distance_row)

    return {"distance_matrix": distance_matrix}
