import math

# Attempt to import required libraries
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csgraph
except ImportError:
    np = None
    sp = None
    csgraph = None


def _mst_using_networkx(num_nodes, edges):
    """Compute MST using NetworkX.

    Args:
        num_nodes (int): Number of nodes.
        edges (list): List of [u, v, weight].

    Returns:
        list: MST edges as [u, v, weight] with u < v.
    """
    if nx is None:
        raise RuntimeError("NetworkX is not available.")
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))
    mst_data = list(nx.minimum_spanning_edges(G, data=True))
    mst = []
    for u, v, data in mst_data:
        if u > v:
            u, v = v, u
        mst.append([u, v, data["weight"]])
    return mst


def _mst_using_scipy(num_nodes, edges):
    """Compute MST using SciPy sparse graph utilities.

    Args:
        num_nodes (int): Number of nodes.
        edges (list): List of [u, v, weight].

    Returns:
        list: MST edges as [u, v, weight] with u < v.
    """
    if sp is None or csgraph is None:
        raise RuntimeError("SciPy sparse modules are not available.")
    # Build symmetric adjacency matrix
    rows = []
    cols = []
    data_vals = []
    for u, v, w in edges:
        rows.append(u)
        cols.append(v)
        data_vals.append(float(w))
        rows.append(v)
        cols.append(u)
        data_vals.append(float(w))
    adj = sp.csr_matrix((data_vals, (rows, cols)), shape=(num_nodes, num_nodes))
    mst_csr = csgraph.minimum_spanning_tree(adj, algorithm="kruskal")
    mst_coo = mst_csr.tocoo()
    mst = []
    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        a, b = (u, v) if u < v else (v, u)
        mst.append([a, b, float(w)])
    return mst


def solve(problem):
    """Solve the Minimum Spanning Tree problem.

    The function selects an algorithm based on the size of the graph:
    * For ``num_nodes`` <= 5000, NetworkX is used (readability and speed for
      moderate sized dense graphs).
    * For larger instances, SciPy's sparse MST implementation is employed to
      keep memory usage low.

    Args:
        problem (dict): Dictionary with keys ``"num_nodes"`` and ``"edges"``.
                        ``edges`` is a list of ``[u, v, weight]`` entries.

    Returns:
        dict: ``{"mst_edges": [...]}`` where the edge list is sorted by the
              tuple ``(u, v)`` for deterministic output.
    """
    num_nodes = problem.get("num_nodes")
    edges = problem.get("edges", [])
    if num_nodes is None:
        raise ValueError("Problem dictionary must contain 'num_nodes'.")

    # Choose algorithm based on node count
    if num_nodes <= 5000:
        mst = _mst_using_networkx(num_nodes, edges)
    else:
        mst = _mst_using_scipy(num_nodes, edges)

    # Ensure deterministic ordering
    mst.sort(key=lambda e: (e[0], e[1]))
    return {"mst_edges": mst}
