import logging
from typing import Any, Dict

# Import cryptography hashing primitives
from cryptography.hazmat.primitives import hashes


def solve(problem: Dict[str, Any]) -> Dict[str, bytes]:
    """Compute the SHA-256 hash of the given plaintext.

    Args:
        problem: A dictionary that must contain the key ``"plaintext"`` whose value
            is a ``bytes`` object.

    Returns:
        A dictionary with a single key ``"digest"`` mapping to the 32‑byte SHA‑256
        digest.
    """
    # Extract plaintext; let any KeyError propagate as it signals a malformed input
    plaintext = problem["plaintext"]

    try:
        # Initialize a SHA256 hash context
        digest = hashes.Hash(hashes.SHA256())
        # Feed the data
        digest.update(plaintext)
        # Finalize and obtain the raw bytes
        hash_value = digest.finalize()
        return {"digest": hash_value}
    except Exception as e:
        logging.error(f"Error during SHA-256 hashing in solve: {e}")
        raise
