import logging
import math
import networkx as nx

class Solver:
    """PageRank solver using NetworkX with specified tolerance and iteration limits."""

    def __init__(self, alpha: float = 0.85, max_iter: int = 1000, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        # Configure logger minimally
        logging.basicConfig(level=logging.ERROR)

    def solve(self, problem: dict) -> dict:
        """Calculate PageRank scores.

        Args:
            problem: {"adjacency_list": [[int, ...], ...]}

        Returns:
            {"pagerank_scores": [float, ...]}
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Edge cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            # Single node graph has PageRank 1.0
            return {"pagerank_scores": [1.0]}

        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                G.add_edge(u, v)

        try:
            pr_dict = nx.pagerank(
                G,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            # Ensure ordering by node index
            pagerank_list = [0.0] * n
            for node, score in pr_dict.items():
                if 0 <= node < n:
                    pagerank_list[node] = float(score)
                else:
                    logging.warning(f"PageRank returned unexpected node {node}")
        except nx.PowerIterationFailedConvergence:
            logging.error(
                f"networkx.pagerank failed to converge after {self.max_iter} iterations."
            )
            pagerank_list = [0.0] * n
        except Exception as e:
            logging.error(f"networkx.pagerank raised an unexpected error: {e}")
            pagerank_list = [0.0] * n

        return {"pagerank_scores": pagerank_list}

# Module-level function for evaluator compatibility
solve = Solver().solve
