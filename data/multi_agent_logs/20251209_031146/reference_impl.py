import math
import logging
import numpy as np
import networkx as nx

class ReferenceSolver:
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-06):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem: dict) -> dict:
        """Calculate PageRank using NetworkX as reference implementation.

        Args:
            problem: {"adjacency_list": List[List[int]]}
        Returns:
            {"pagerank_scores": List[float]}
        """
        adj_list = problem.get("adjacency_list")
        if adj_list is None:
            raise ValueError("Problem missing 'adjacency_list'")
        n = len(adj_list)
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                G.add_edge(u, v)
        try:
            pr_dict = nx.pagerank(G, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        except nx.PowerIterationFailedConvergence:
            logging.error("NetworkX pagerank failed to converge")
            return {"pagerank_scores": [0.0] * n}
        # Convert dict to list ordered by index
        pagerank_list = [float(pr_dict.get(i, 0.0)) for i in range(n)]
        # Normalize to sum 1 (networkx already does, but guard against tiny drift)
        total = sum(pagerank_list)
        if total > 0:
            pagerank_list = [v / total for v in pagerank_list]
        return {"pagerank_scores": pagerank_list}

# expose top-level function as required by harness
solver = ReferenceSolver()
def solve(problem: dict) -> dict:
    return solver.solve(problem)
