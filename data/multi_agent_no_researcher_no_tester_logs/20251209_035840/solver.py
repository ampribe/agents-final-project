import sys
import json
from typing import Any, Dict, List


def solve(problem: Dict[str, Any]) -> Dict[str, List[int]]:
    """Solve the Stable Matching problem using the Gale–Shapley algorithm.

    Args:
        problem: A dictionary with keys ``"proposer_prefs"`` and ``"receiver_prefs"``.
                 Each maps to a list of lists (or a dict) where the inner list contains
                 the ranked indices of the opposite side.

    Returns:
        A dictionary ``{"matching": matching}`` where ``matching[i]`` is the index of
        the receiver matched to proposer ``i``.
    """
    # Extract raw preference data
    prop_raw = problem["proposer_prefs"]
    recv_raw = problem["receiver_prefs"]

    # Normalise to list-of-lists (support dict representation as in validator)
    if isinstance(prop_raw, dict):
        n = len(prop_raw)
        proposer_prefs: List[List[int]] = [prop_raw[i] for i in range(n)]
    else:
        proposer_prefs = list(prop_raw)
        n = len(proposer_prefs)

    if isinstance(recv_raw, dict):
        receiver_prefs: List[List[int]] = [recv_raw[i] for i in range(n)]
    else:
        receiver_prefs = list(recv_raw)

    # Pre‑compute receiver ranking tables for O(1) comparison
    recv_rank: List[List[int]] = [[0] * n for _ in range(n)]
    for r, prefs in enumerate(receiver_prefs):
        for rank, p in enumerate(prefs):
            recv_rank[r][p] = rank

    # Gale‑Shapley core
    next_prop: List[int] = [0] * n            # next receiver to propose to for each proposer
    recv_match: List[int | None] = [None] * n  # current proposer matched to each receiver
    free: List[int] = list(range(n))           # proposers that are currently free

    while free:
        p = free.pop(0)  # take the first free proposer
        # Propose to the next receiver on p's list
        r = proposer_prefs[p][next_prop[p]]
        next_prop[p] += 1

        current = recv_match[r]
        if current is None:
            # Receiver is free, accept proposal
            recv_match[r] = p
        else:
            # Receiver chooses the better proposer according to her ranking
            if recv_rank[r][p] < recv_rank[r][current]:
                # She prefers the new proposer
                recv_match[r] = p
                free.append(current)  # the dumped proposer becomes free again
            else:
                # She rejects the new proposer
                free.append(p)

    # Build proposer -> receiver mapping
    matching: List[int] = [0] * n
    for r, p in enumerate(recv_match):
        # recv_match is guaranteed to contain a proposer for each receiver
        matching[p] = r  # type: ignore[arg-type]

    return {"matching": matching}


if __name__ == "__main__":
    # When run as a script, read a JSON object from stdin and output the result.
    input_data = sys.stdin.read()
    if not input_data.strip():
        print(json.dumps({"error": "No input provided"}))
        sys.exit(1)
    problem = json.loads(input_data)
    result = solve(problem)
    print(json.dumps(result))
