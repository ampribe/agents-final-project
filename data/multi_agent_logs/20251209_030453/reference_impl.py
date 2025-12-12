import networkx as nx

def solve(problem: dict) -> dict:
    """Reference solution for Minimum Spanning Tree using NetworkX.
    Returns edges of the MST (or forest for disconnected graphs) sorted by (u, v).
    """
    G = nx.Graph()
    num_nodes = problem["num_nodes"]
    edges = problem["edges"]
    G.add_nodes_from(range(num_nodes))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    mst_edges_data = list(nx.minimum_spanning_edges(G, data=True))
    mst_edges = []
    for u, v, data in mst_edges_data:
        if u > v:
            u, v = v, u
        mst_edges.append([u, v, data["weight"]])
    mst_edges.sort(key=lambda x: (x[0], x[1]))
    return {"mst_edges": mst_edges}
