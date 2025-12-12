import math
import logging
import numpy as np
import networkx as nx

class ReferencePagerank:
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        # configure logger minimally
        logging.basicConfig(level=logging.ERROR)

    def solve(self, problem: dict) -> dict:
        """Calculate PageRank scores using NetworkX, matching the reference spec."""
        adj_list = problem.get("adjacency_list", [])
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
            pagerank_dict = nx.pagerank(G, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
            scores = [0.0] * n
            for node, score in pagerank_dict.items():
                if 0 <= node < n:
                    scores[node] = float(score)
        except nx.PowerIterationFailedConvergence:
            logging.error("networkx.pagerank failed to converge after %d iterations.", self.max_iter)
            scores = [0.0] * n
        except Exception as e:
            logging.error("networkx.pagerank raised an unexpected error: %s", e)
            scores = [0.0] * n
        return {"pagerank_scores": scores}

# expose a simple function for harness import
_ref = ReferencePagerank()

def solve(problem: dict) -> dict:
    return _ref.solve(problem)
