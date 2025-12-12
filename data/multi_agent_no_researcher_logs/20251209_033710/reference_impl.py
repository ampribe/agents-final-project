# reference_impl.py
import sys
from typing import Any, Dict, List

def solve(problem: Dict[str, Any]) -> Dict[str, List[int]]:
    prop_raw = problem["proposer_prefs"]
    recv_raw = problem["receiver_prefs"]

    # Normalise to list-of-lists
    if isinstance(prop_raw, dict):
        n = len(prop_raw)
        proposer_prefs = [prop_raw[i] for i in range(n)]
    else:
        proposer_prefs = list(prop_raw)
        n = len(proposer_prefs)

    if isinstance(recv_raw, dict):
        receiver_prefs = [recv_raw[i] for i in range(n)]
    else:
        receiver_prefs = list(recv_raw)

    # Build receiver ranking table for O(1) comparisons
    recv_rank = [[0] * n for _ in range(n)]
    for r, prefs in enumerate(receiver_prefs):
        for rank, p in enumerate(prefs):
            recv_rank[r][p] = rank

    next_prop = [0] * n          # next receiver to propose to for each proposer
    recv_match: List[Any] = [None] * n  # current proposer matched to each receiver
    free = list(range(n))        # stack/queue of free proposers

    while free:
        p = free.pop(0)  # FIFO order (could be stack as well)
        r = proposer_prefs[p][next_prop[p]]
        next_prop[p] += 1

        cur = recv_match[r]
        if cur is None:
            recv_match[r] = p
        else:
            # Receiver prefers lower rank value
            if recv_rank[r][p] < recv_rank[r][cur]:
                recv_match[r] = p
                free.append(cur)
            else:
                free.append(p)

    # Convert receiver->proposer mapping to proposer->receiver list
    matching = [0] * n
    for r, p in enumerate(recv_match):
        matching[p] = r

    return {"matching": matching}

if __name__ == "__main__":
    # Simple manual test if run directly
    import json
    problem = json.load(sys.stdin)
    result = solve(problem)
    json.dump(result, sys.stdout)
