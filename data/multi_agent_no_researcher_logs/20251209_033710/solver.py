import logging
from typing import Any, Dict, List


def solve(problem: Dict[str, Any]) -> Dict[str, List[int]]:
    """Solve the stable matching problem using proposer-optimal Gale‑Shapley.

    Args:
        problem: A dictionary with keys "proposer_prefs" and "receiver_prefs".
                 Each value can be a list of lists or a dict mapping indices to
                 preference lists.

    Returns:
        A dictionary with a single key "matching" mapping each proposer index
        to the index of the receiver they are matched with.
    """
    # Extract raw preference data
    prop_raw = problem.get("proposer_prefs")
    recv_raw = problem.get("receiver_prefs")
    if prop_raw is None or recv_raw is None:
        raise ValueError("Problem must contain 'proposer_prefs' and 'receiver_prefs'.")

    # Normalise proposer preferences to list of lists
    if isinstance(prop_raw, dict):
        n = len(prop_raw)
        proposer_prefs: List[List[int]] = [prop_raw[i] for i in range(n)]
    else:
        proposer_prefs = list(prop_raw)
        n = len(proposer_prefs)

    # Normalise receiver preferences to list of lists
    if isinstance(recv_raw, dict):
        receiver_prefs: List[List[int]] = [recv_raw[i] for i in range(n)]
    else:
        receiver_prefs = list(recv_raw)

    # Basic sanity checks
    if n == 0:
        return {"matching": []}
    if any(len(p) != n for p in proposer_prefs):
        raise ValueError("Each proposer preference list must contain n entries.")
    if any(len(r) != n for r in receiver_prefs):
        raise ValueError("Each receiver preference list must contain n entries.")

    # Pre‑compute receiver ranking tables for O(1) comparisons
    recv_rank: List[List[int]] = [[0] * n for _ in range(n)]
    for r, prefs in enumerate(receiver_prefs):
        for rank, p in enumerate(prefs):
            recv_rank[r][p] = rank

    # Gale‑Shapley algorithm
    next_prop = [0] * n  # next receiver index each proposer will propose to
    recv_match: List[Any] = [None] * n  # current proposer matched to each receiver
    free = list(range(n))  # proposers that are currently free

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
            # Receiver chooses the better proposer according to its ranking
            if recv_rank[r][p] < recv_rank[r][current]:
                # New proposer is preferred
                recv_match[r] = p
                free.append(current)  # former partner becomes free
            else:
                # Receiver rejects the proposal
                free.append(p)

    # Build proposer -> receiver matching list
    matching = [0] * n
    for r, p in enumerate(recv_match):
        matching[p] = r

    return {"matching": matching}

# If the module is executed directly, run a quick sanity test
if __name__ == "__main__":
    example = {
        "proposer_prefs": [
            [0, 1, 2],
            [1, 0, 2],
            [0, 1, 2],
        ],
        "receiver_prefs": [
            [1, 0, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
    }
    print(solve(example))
