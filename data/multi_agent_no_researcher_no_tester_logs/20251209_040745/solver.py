import numpy as np

def solve(problem: dict) -> dict:
    """Compute PageRank scores for a directed graph.

    The graph is given by an adjacency list where adjacency_list[i] contains the
    sorted list of nodes that node *i* points to (outgoing edges). The function
    returns a dictionary with a single key ``"pagerank_scores"`` mapping to a
    list of floats representing the PageRank values for each node.
    """
    adj = problem.get("adjacency_list", [])
    n = len(adj)
    # Edge cases
    if n == 0:
        return {"pagerank_scores": []}
    if n == 1:
        return {"pagerank_scores": [1.0]}

    alpha = 0.85
    epsilon = 1e-10
    max_iter = 100

    # Out-degree for each node
    out_deg = np.array([len(neigh) for neigh in adj], dtype=np.int64)
    dangling = out_deg == 0

    # Initialize rank vector uniformly
    r = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(max_iter):
        r_new = np.zeros(n, dtype=np.float64)
        # Distribute rank from non‑dangling nodes
        for i, neighbors in enumerate(adj):
            d = out_deg[i]
            if d > 0:
                contrib = r[i] / d
                for j in neighbors:
                    r_new[j] += contrib
        # Add contribution from dangling nodes (equally to all nodes)
        dangling_sum = r[dangling].sum()
        if dangling_sum != 0.0:
            r_new += dangling_sum / n
        # Apply damping factor and teleportation
        r_new = alpha * r_new + (1.0 - alpha) / n
        # Check convergence (L1 norm)
        if np.abs(r_new - r).sum() < epsilon:
            r = r_new
            break
        r = r_new

    # Normalize to ensure sum equals 1 (protect against numerical drift)
    total = r.sum()
    if total != 0.0:
        r = r / total

    return {"pagerank_scores": r.tolist()}
