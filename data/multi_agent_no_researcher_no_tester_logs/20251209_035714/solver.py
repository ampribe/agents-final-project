# solver.py
"""All-Pairs Shortest Paths (Dijkstra) Solver

The solver expects a problem dictionary describing a weighted, undirected graph in
CSR (Compressed Sparse Row) format with the keys:

* ``data`` – list of edge weights (non‑negative floats)
* ``indices`` – list of column indices for each weight
* ``indptr`` – list of pointers to the start of each row in ``data``/``indices``
* ``shape`` – ``[n, n]`` where ``n`` is the number of nodes

It returns a dictionary with a single key ``"distance_matrix"`` whose value is a
list of ``n`` lists. ``distance_matrix[i][j]`` holds the shortest path length from
node ``i`` to node ``j`` or ``None`` when ``j`` is unreachable from ``i``.

The implementation converts the CSR representation to an **undirected**
adjacency list for fast neighbour look‑ups and runs Dijkstra's algorithm from
every source node.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional


def _validate_csr(problem: Dict[str, Any]) -> Optional[str]:
    """Validate the CSR structure.

    Returns ``None`` if everything looks fine, otherwise an error message.
    """
    required_keys = {"data", "indices", "indptr", "shape"}
    missing = required_keys - problem.keys()
    if missing:
        return f"Missing CSR keys: {missing}"

    shape = problem["shape"]
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        return "'shape' must be a list or tuple of length 2"
    n_rows, n_cols = shape
    if n_rows != n_cols:
        return "Graph must be square (undirected)"
    n = n_rows

    data = problem["data"]
    indices = problem["indices"]
    indptr = problem["indptr"]

    if len(indptr) != n + 1:
        return f"'indptr' length must be n+1 ({n+1}), got {len(indptr)}"
    if len(data) != len(indices):
        return "'data' and 'indices' must have the same length"
    if any(not (0 <= idx < n) for idx in indices):
        return "'indices' contains out‑of‑range column indices"
    if any(indptr[i] > indptr[i + 1] for i in range(len(indptr) - 1)):
        return "'indptr' must be non‑decreasing"
    if indptr[0] != 0 or indptr[-1] != len(data):
        return "'indptr' must start at 0 and end at len(data)"

    return None


def _csr_to_adjlist(data: List[float], indices: List[int], indptr: List[int]) -> List[List[tuple[int, float]]]:
    """Convert CSR to an undirected adjacency list.

    For each directed entry (u → v, w) we also add the reverse edge (v → u, w).
    Duplicate entries are allowed – Dijkstra's algorithm works correctly with
    parallel edges, keeping the smallest distance.
    """
    n = len(indptr) - 1
    adj: List[List[tuple[int, float]]] = [[] for _ in range(n)]
    for u in range(n):
        start = indptr[u]
        end = indptr[u + 1]
        for idx in range(start, end):
            v = indices[idx]
            w = data[idx]
            adj[u].append((v, w))
            if u != v:
                adj[v].append((u, w))
    return adj


def _dijkstra(adj: List[List[tuple[int, float]]], source: int) -> List[float]:
    """Single‑source Dijkstra using a binary heap.

    ``adj`` is an adjacency list where ``adj[u]`` is a list of ``(v, weight)``.
    Returns a list of distances from ``source`` to every node; unreachable
    vertices have ``math.inf``.
    """
    n = len(adj)
    dist = [math.inf] * n
    dist[source] = 0.0
    heap: List[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue  # stale entry
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Solve the all‑pairs shortest path problem.

    Parameters
    ----------
    problem: dict
        CSR representation of the graph.

    Returns
    -------
    dict
        ``{"distance_matrix": …}`` where the matrix is a list of lists of floats
        or ``None`` for unreachable pairs.
    """
    err = _validate_csr(problem)
    if err is not None:
        # Return empty matrix – validator will treat this as a failure case.
        return {"distance_matrix": []}

    data: List[float] = problem["data"]
    indices: List[int] = problem["indices"]
    indptr: List[int] = problem["indptr"]
    n = problem["shape"][0]

    adj = _csr_to_adjlist(data, indices, indptr)

    distance_matrix: List[List[Optional[float]]] = []
    for src in range(n):
        dist = _dijkstra(adj, src)
        row = [None if math.isinf(d) else d for d in dist]
        distance_matrix.append(row)

    return {"distance_matrix": distance_matrix}


__all__ = ["solve"]
