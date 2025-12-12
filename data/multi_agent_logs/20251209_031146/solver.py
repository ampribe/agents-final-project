import sys
import json
import math
import numpy as np
import scipy.sparse as sp

def _pagerank(adjacency_list, alpha=0.85, tol=1e-12, max_iter=1000):
    """Compute PageRank scores for a directed graph.

    Parameters
    ----------
    adjacency_list : List[List[int]]
        adjacency_list[i] contains the nodes that i points to.
    alpha : float, optional
        Damping factor (default 0.85).
    tol : float, optional
        Convergence tolerance for L1 norm change (default 1e-12).
    max_iter : int, optional
        Maximum number of power‑iteration steps (default 1000).

    Returns
    -------
    List[float]
        PageRank scores that sum to 1.
    """
    n = len(adjacency_list)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Build column‑stochastic matrix P (in CSC format for efficient column access)
    data = []
    rows = []
    cols = []
    dangling = np.zeros(n, dtype=bool)
    for i, neighbors in enumerate(adjacency_list):
        d = len(neighbors)
        if d == 0:
            dangling[i] = True
            continue
        weight = 1.0 / d
        for j in neighbors:
            rows.append(j)
            cols.append(i)
            data.append(weight)
    P = sp.csc_matrix((data, (rows, cols)), shape=(n, n))

    # Initial uniform distribution
    r = np.full(n, 1.0 / n, dtype=float)
    teleport = (1.0 - alpha) / n

    for _ in range(max_iter):
        r_last = r.copy()
        # Sparse matrix multiplication P @ r_last
        pr = P.dot(r_last)
        # Add contribution from dangling nodes (they link to all nodes uniformly)
        dangling_sum = r_last[dangling].sum()
        r = alpha * (pr + dangling_sum / n) + teleport
        # Convergence check (L1 norm)
        if np.abs(r - r_last).sum() < tol:
            break

    # Ensure exact normalization (numerical errors may accumulate)
    r_sum = r.sum()
    if not math.isclose(r_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        r = r / r_sum
    return r.tolist()

def solve(problem):
    """Entry point used by the evaluator.

    Parameters
    ----------
    problem : dict
        Must contain the key "adjacency_list".

    Returns
    -------
    dict
        {"pagerank_scores": [...]} where the list length equals the number of nodes.
    """
    adjacency_list = problem.get("adjacency_list", [])
    # Validate basic structure – the validator will perform strict checks later.
    scores = _pagerank(adjacency_list, alpha=0.85, tol=1e-12, max_iter=1000)
    return {"pagerank_scores": scores}

if __name__ == "__main__":
    # Read JSON from stdin, write JSON to stdout.
    data = sys.stdin.read()
    if not data:
        sys.exit(0)
    problem = json.loads(data)
    solution = solve(problem)
    json.dump(solution, sys.stdout, separators=(",", ":"))
