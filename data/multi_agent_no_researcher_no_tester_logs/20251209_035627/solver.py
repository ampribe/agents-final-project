# solver.py
"""Minimum Spanning Tree solver using Kruskal's algorithm.

The problem input is a dictionary with keys:
  - "num_nodes": int, number of nodes (0-indexed)
  - "edges": list of [u, v, weight] where u and v are node indices and weight is a float.

The solver returns a dictionary with key:
  - "mst_edges": list of edges in the MST, each as [u, v, weight],
    sorted in ascending order by (u, v) for deterministic output.
"""

from __future__ import annotations

import json
import sys
from typing import Any, List, Tuple


class UnionFind:
    """Disjoint Set Union (Union‑Find) with path compression and union by rank."""

    def __init__(self, size: int):
        self.parent: List[int] = list(range(size))
        self.rank: List[int] = [0] * size

    def find(self, x: int) -> int:
        """Find the root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union the sets containing x and y.

        Returns True if a union was performed (i.e., x and y were in different sets),
        otherwise False.
        """
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False
        # union by rank
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


def solve(problem: dict[str, Any]) -> dict[str, List[List[Any]]]:
    """Compute the Minimum Spanning Tree (MST) of an undirected weighted graph.

    Parameters
    ----------
    problem: dict
        A dictionary containing:
        - "num_nodes": int – number of vertices (identified by 0 … num_nodes‑1).
        - "edges": list of [u, v, weight] – undirected edges with a float weight.

    Returns
    -------
    dict
        A dictionary with a single key "mst_edges" → list of edges belonging to the MST.
        Each edge is represented as [u, v, weight] where u < v.
        The list is sorted lexicographically by (u, v).
    """
    # Basic validation
    if "num_nodes" not in problem or "edges" not in problem:
        raise ValueError("Problem must contain 'num_nodes' and 'edges'.")

    num_nodes: int = int(problem["num_nodes"])
    raw_edges: List[Tuple[int, int, float]] = []
    for edge in problem["edges"]:
        if len(edge) != 3:
            raise ValueError("Each edge must be a list of three elements: [u, v, weight].")
        u, v, w = edge
        u = int(u)
        v = int(v)
        w = float(w)
        raw_edges.append((u, v, w))

    # Sort edges by weight (ascending) for Kruskal
    sorted_edges = sorted(raw_edges, key=lambda e: e[2])

    uf = UnionFind(num_nodes)
    mst_edges: List[Tuple[int, int, float]] = []

    for u, v, w in sorted_edges:
        if uf.union(u, v):
            # Ensure deterministic orientation: store with smaller node first
            if u > v:
                u, v = v, u
            mst_edges.append((u, v, w))
            if len(mst_edges) == num_nodes - 1:
                break

    # Final sorting by (u, v) for output consistency
    mst_edges.sort(key=lambda e: (e[0], e[1]))

    # Convert to the required list-of-lists format, keeping weight as float
    result_edges = [[u, v, w] for u, v, w in mst_edges]
    return {"mst_edges": result_edges}


if __name__ == "__main__":
    # Allow running the solver from the command line with JSON input on stdin.
    # Example: echo '{"num_nodes":5,"edges":[[0,1,1.2],[0,2,2.3],[1,2,1.0],[2,3,3.4],[1,4,0.9]]}' | .venv/bin/python solver.py
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Failed to parse JSON input: {exc}\n")
        sys.exit(1)

    output = solve(input_data)
    json.dump(output, sys.stdout)
    sys.stdout.flush()
