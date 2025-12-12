import typing
from typing import List, Tuple, Dict, Any

class UnionFind:
    def __init__(self, size: int):
        self.parent: List[int] = list(range(size))
        self.rank: List[int] = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True


def solve(problem: Dict[str, Any]) -> Dict[str, List[List[Any]]]:
    """Compute the Minimum Spanning Tree (MST) of an undirected weighted graph.

    Args:
        problem: A dictionary with keys:
            - "num_nodes": int, number of nodes (0 .. num_nodes-1)
            - "edges": List[[u, v, weight]], edge list where weight is a float

    Returns:
        A dictionary with a single key "mst_edges" containing a list of edges
        that form the MST. Each edge is represented as [u, v, weight] with
        u <= v, and the list is sorted ascending by (u, v) for deterministic
        output.
    """
    num_nodes: int = problem["num_nodes"]
    raw_edges: List[Tuple[int, int, float]] = [tuple(e) for e in problem.get("edges", [])]

    # Sort edges by weight, then by node indices for deterministic tie-breaking
    sorted_edges = sorted(raw_edges, key=lambda e: (e[2], e[0], e[1]))

    uf = UnionFind(num_nodes)
    mst: List[List[Any]] = []

    for u, v, w in sorted_edges:
        if uf.union(u, v):
            if u > v:
                u, v = v, u
            mst.append([u, v, w])
            if len(mst) == num_nodes - 1:
                break

    mst.sort(key=lambda edge: (edge[0], edge[1]))
    return {"mst_edges": mst}
