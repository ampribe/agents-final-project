# -*- coding: utf-8 -*-
"""Minimum Spanning Tree Solver.

This module provides a single ``solve`` function that receives a problem
definition – a dictionary containing the number of nodes and a list of
edges – and returns a dictionary with the edges that form a minimum
spanning tree (MST) of the undirected weighted graph.

The implementation follows the reference solution from ``REFERENCE.md``
and uses the ``networkx`` library to compute the MST via
``networkx.minimum_spanning_edges``.  The returned edges are normalised so
that ``u < v`` for each edge and then sorted by ``(u, v)`` to guarantee a
deterministic order that matches the validation logic in ``VALIDATION.md``.

The expected input format is::

    {
        "num_nodes": int,
        "edges": [[u, v, weight], ...]
    }

where ``u`` and ``v`` are integer node identifiers in the range
``[0, num_nodes-1]`` and ``weight`` is a numeric value (float or int).

The output format is a dictionary with a single key ``"mst_edges"`` whose
value is a list of edges in the same ``[u, v, weight]`` representation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx


def solve(problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
    """Compute the Minimum Spanning Tree of an undirected weighted graph.

    Parameters
    ----------
    problem: dict
        A dictionary with the keys ``"num_nodes"`` (int) and ``"edges"``
        (list of ``[u, v, weight]``). ``weight`` can be any numeric type.

    Returns
    -------
    dict
        ``{"mst_edges": [[u, v, weight], ...]}`` where the edges are sorted
        by ``(u, v)`` and ``u < v`` for each edge.
    """

    # Build the graph.
    G = nx.Graph()
    num_nodes = problem.get("num_nodes")
    edges = problem.get("edges", [])

    # Ensure all nodes are present, even isolated ones.
    G.add_nodes_from(range(num_nodes))

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Compute the MST.
    mst_iter = nx.minimum_spanning_edges(G, data=True)
    mst_edges: List[List[float]] = []
    for u, v, data in mst_iter:
        if u > v:
            u, v = v, u
        mst_edges.append([u, v, data["weight"]])

    # Sort for deterministic output.
    mst_edges.sort(key=lambda e: (e[0], e[1]))
    return {"mst_edges": mst_edges}


if __name__ == "__main__":
    example = {
        "num_nodes": 5,
        "edges": [
            [0, 1, 1.2],
            [0, 2, 2.3],
            [1, 2, 1.0],
            [2, 3, 3.4],
            [1, 4, 0.9],
        ],
    }
    print(solve(example))
