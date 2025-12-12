import logging
from typing import Any, Dict, List


def solve(problem: Dict[str, Any]) -> Dict[str, List[int]]:
    """Solve the stable matching problem using the Gale‑Shapley algorithm.

    Args:
        problem: A dictionary with keys "proposer_prefs" and "receiver_prefs".
                 Each value may be a list of lists or a dict mapping indices to
                 preference lists.

    Returns:
        A dictionary with a single key "matching" whose value is a list where
        ``matching[p]`` is the index of the receiver matched to proposer ``p``.
    """
    # Extract raw preference data
    prop_raw = problem.get("proposer_prefs")
    recv_raw = problem.get("receiver_prefs")

    # Normalise proposer preferences to a list of lists
    if isinstance(prop_raw, dict):
        n = len(prop_raw)
        proposer_prefs: List[List[int]] = [prop_raw[i] for i in range(n)]
    else:
        proposer_prefs = list(prop_raw)
        n = len(proposer_prefs)

    # Normalise receiver preferences to a list of lists
    if isinstance(recv_raw, dict):
        receiver_prefs: List[List[int]] = [recv_raw[i] for i in range(n)]
    else:
        receiver_prefs = list(recv_raw)

    # Pre‑compute receiver ranking for O(1) comparisons
    # recv_rank[r][p] gives the rank (lower is better) of proposer p in receiver r's list
    recv_rank: List[List[int]] = [[0] * n for _ in range(n)]
    for r, prefs in enumerate(receiver_prefs):
        for rank, p in enumerate(prefs):
            recv_rank[r][p] = rank

    # next_prop[p] points to the next receiver in p's list to propose to
    next_prop: List[int] = [0] * n
    # recv_match[r] holds the current proposer matched to receiver r (or None)
    recv_match: List[Any] = [None] * n
    # Initially all proposers are free
    free_proposers: List[int] = list(range(n))

    while free_proposers:
        p = free_proposers.pop(0)
        # Propose to the next receiver on p's list
        r = proposer_prefs[p][next_prop[p]]
        next_prop[p] += 1

        current = recv_match[r]
        if current is None:
            # Receiver r is free, accept proposal
            recv_match[r] = p
        else:
            # Receiver r already matched; keep the better proposer according to her preferences
            if recv_rank[r][p] < recv_rank[r][current]:
                # r prefers new proposer p
                recv_match[r] = p
                free_proposers.append(current)  # the dumped proposer becomes free again
            else:
                # r rejects p; p stays free and will propose to the next receiver later
                free_proposers.append(p)

    # Build proposer -> receiver mapping
    matching: List[int] = [0] * n
    for r, p in enumerate(recv_match):
        matching[p] = r

    return {"matching": matching}

# Export a callable ``solve`` at module import time for convenience
# (some evaluation harnesses may import ``solver`` and call ``solver.solve`` directly)

if __name__ == "__main__":
    # Simple self‑test when run as a script
    example = {
        "proposer_prefs": [[0, 1, 2], [1, 0, 2], [0, 1, 2]],
        "receiver_prefs": [[1, 0, 2], [0, 1, 2], [0, 1, 2]],
    }
    print(solve(example))
