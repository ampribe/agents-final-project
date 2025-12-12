# Solver for Base64 Encoding Task

import base64
import logging
from typing import Any, Dict


def solve(problem: Dict[str, Any]) -> Dict[str, bytes]:
    """Encode the plaintext using the standard Base64 algorithm.

    Args:
        problem (dict): Dictionary containing the key "plaintext" with bytes data.

    Returns:
        dict: Dictionary with a single key "encoded_data" mapping to the Base64-encoded bytes.
    """
    # Extract plaintext; let any KeyError propagate as it indicates malformed input.
    plaintext = problem["plaintext"]
    try:
        # Encode using Python's built-in base64 module.
        encoded_data = base64.b64encode(plaintext)
        return {"encoded_data": encoded_data}
    except Exception as e:
        logging.error(f"Error during Base64 encoding in solve: {e}")
        raise
