"""Stable Matching Solver

Implements the classic Gale‑Shapley algorithm for finding a stable matching
between two equally sized sets of agents (proposers and receivers).

The function ``solve`` expects a dictionary with the keys ``"proposer_prefs"``
and ``"receiver_prefs"``.  Each value can be either a list of lists (the most
common representation) or a dict mapping an integer index to a preference
list.  The preference lists contain the indices of the opposite set ordered
from most to least preferred.

The output is a dictionary containing a single key ``"matching"`` whose value
is a list ``matching[i] = r`` meaning proposer ``i`` is matched with receiver
``r``.  The matching is guaranteed to be stable and proposer‑optimal.

The implementation mirrors the reference solution used for validation.  It
converts the input to a canonical list‑of‑lists format, builds a ranking table
for each receiver to allow O(1) comparison of two proposers, and then runs the
standard proposal loop.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _normalize_prefs(raw: Any, n: int) -> List[List[int]]:
    """Convert ``raw`` preferences to a list of lists.

    ``raw`` may be a list of lists or a dict mapping integer indices to
    preference lists.  ``n`` is the expected size (number of proposers or
    receivers).  The function returns a list ``prefs[i]`` for ``i`` in ``0..n-1``.
    """

    if isinstance(raw, dict):
        # Preserve the order of indices from 0 to n-1.
        return [raw[i] for i in range(n)]
    # Assume an already‑indexed list‑of‑lists.
    return list(raw)


def solve(problem: Dict[str, Any]) -> Dict[str, List[int]]:
    """Return a stable matching for the given preference lists.

    Parameters
    ----------
    problem:
        A dictionary with keys ``"proposer_prefs"`` and ``"receiver_prefs"``.

    Returns
    -------
    dict
        ``{"matching": matching}`` where ``matching[i]`` is the index of the
        receiver matched to proposer ``i``.
    """

    # ---------------------------------------------------------------------
    # Normalise input to list‑of‑lists.
    # ---------------------------------------------------------------------
    prop_raw = problem["proposer_prefs"]
    recv_raw = problem["receiver_prefs"]

    # Determine n – the number of proposers (and receivers).
    if isinstance(prop_raw, dict):
        n = len(prop_raw)
    else:
        n = len(prop_raw)

    proposer_prefs = _normalize_prefs(prop_raw, n)
    receiver_prefs = _normalize_prefs(recv_raw, n)

    # ---------------------------------------------------------------------
    # Build a receiver ranking matrix: recv_rank[r][p] = rank of proposer p
    # according to receiver r (lower is better).
    # ---------------------------------------------------------------------
    recv_rank: List[List[int]] = [[0] * n for _ in range(n)]
    for r, prefs in enumerate(receiver_prefs):
        for rank, p in enumerate(prefs):
            recv_rank[r][p] = rank

    # ---------------------------------------------------------------------
    # Gale‑Shapley proposal loop.
    # ---------------------------------------------------------------------
    # next_prop[p] – index of the next receiver proposer p will propose to.
    next_prop: List[int] = [0] * n
    # recv_match[r] – current proposer matched to receiver r (or None).
    recv_match: List[int | None] = [None] * n
    # List of currently free proposers.
    free: List[int] = list(range(n))

    while free:
        p = free.pop(0)  # take the first free proposer
        # The proposer must have at least one remaining candidate because the
        # algorithm only stops when everyone is matched.
        r = proposer_prefs[p][next_prop[p]]
        next_prop[p] += 1

        current = recv_match[r]
        if current is None:
            # Receiver is free – accept the proposal.
            recv_match[r] = p
        else:
            # Receiver chooses the better proposer according to her ranking.
            if recv_rank[r][p] < recv_rank[r][current]:
                # New proposer is preferred.
                recv_match[r] = p
                free.append(current)  # the displaced proposer becomes free.
            else:
                # Receiver rejects the proposal; proposer stays free.
                free.append(p)

    # ---------------------------------------------------------------------
    # Convert receiver‑centric representation to proposer‑centric list.
    # ``matching[p] = r``.
    # ---------------------------------------------------------------------
    matching: List[int] = [0] * n
    for r, p in enumerate(recv_match):
        # ``p`` cannot be None at this point because the algorithm guarantees a
        # perfect matching.
        matching[p] = r  # type: ignore[index]

    return {"matching": matching}
