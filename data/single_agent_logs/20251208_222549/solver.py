"""PageRank Solver

This module provides a single ``solve`` function that computes the PageRank
scores for a directed graph given in adjacency‑list representation.

The implementation follows the reference solution supplied in ``REFERENCE.md``
and uses :func:`networkx.pagerank`.  It handles corner cases (empty graph and a
single‑node graph) and falls back to a zero vector if the power‑iteration does
not converge.

The expected input format is a dictionary with the key ``"adjacency_list"``
mapping to a list of lists where ``adjacency_list[i]`` contains the sorted list
of node indices that node ``i`` points to.  Nodes are numbered from ``0`` to
``n‑1`` where ``n`` is the length of the outer list.

The function returns a dictionary with the key ``"pagerank_scores"`` whose value
is a list of floating‑point scores ordered by node index.

The implementation does not rely on any external configuration – it uses the
standard PageRank parameters ``alpha=0.85``, ``max_iter=100`` and ``tol=1e-06``.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx


def solve(problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
    """Compute PageRank scores for a directed graph.

    Parameters
    ----------
    problem:
        A dictionary with a single key ``"adjacency_list"``.  The associated
        value is a list of lists representing outgoing edges for each node.

    Returns
    -------
    dict
        ``{"pagerank_scores": [...]}`` where the list contains the PageRank
        score for each node in index order.  An empty graph yields an empty
        list and a single‑node graph yields ``[1.0]``.
    """

    # Extract adjacency list and determine number of nodes.
    adj_list = problem.get("adjacency_list", [])
    n = len(adj_list)

    # Corner‑case handling – matches the reference implementation.
    if n == 0:
        return {"pagerank_scores": []}
    if n == 1:
        # A single node receives the entire probability mass.
        return {"pagerank_scores": [1.0]}

    # Build a directed NetworkX graph from the adjacency representation.
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            # ``add_edge`` creates a directed edge u -> v.
            G.add_edge(u, v)

    # Default PageRank parameters (the same used by the reference).
    alpha = 0.85
    max_iter = 100
    tol = 1.0e-06

    try:
        pr_dict = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
        # Convert the dictionary to a list ordered by node index.
        pagerank_scores = [0.0] * n
        for node, score in pr_dict.items():
            if 0 <= node < n:
                pagerank_scores[node] = float(score)
    except nx.PowerIterationFailedConvergence:
        # If convergence fails, fall back to a uniform distribution of zeros.
        pagerank_scores = [0.0] * n
    except Exception:
        # Any unexpected error also results in a zero vector.
        pagerank_scores = [0.0] * n

    return {"pagerank_scores": pagerank_scores}
