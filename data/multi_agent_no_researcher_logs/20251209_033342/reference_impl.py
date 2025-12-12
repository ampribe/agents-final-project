import sys

def solve(problem):
    """Deterministic Kruskal MST implementation.
    Returns edges as list of [u, v, weight] with u < v, sorted by (u, v).
    """
    num_nodes = problem["num_nodes"]
    edges = problem["edges"]
    # sort edges by weight, then by (u, v) to ensure deterministic tie-breaking
    sorted_edges = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
    parent = list(range(num_nodes))
    rank = [0] * num_nodes
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        xr, yr = find(x), find(y)
        if xr == yr:
            return False
        if rank[xr] < rank[yr]:
            parent[xr] = yr
        elif rank[xr] > rank[yr]:
            parent[yr] = xr
        else:
            parent[yr] = xr
            rank[xr] += 1
        return True
    mst = []
    for u, v, w in sorted_edges:
        if union(u, v):
            if u > v:
                u, v = v, u
            mst.append([u, v, w])
            if len(mst) == num_nodes - 1:
                break
    # sort final list by (u, v)
    mst.sort(key=lambda x: (x[0], x[1]))
    return {"mst_edges": mst}
