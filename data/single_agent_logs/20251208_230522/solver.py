# -*- coding: utf-8 -*-
"""AES-GCM Encryption Solver

This module provides a ``solve`` function that encrypts a plaintext using
AES in Galois/Counter Mode (GCM). The implementation follows the reference
provided in ``REFERENCE.md`` and is compatible with the validation logic in
``VALIDATION.md``.

The function expects a ``problem`` dictionary with the following keys:

* ``key`` (bytes): AES key – must be 16, 24, or 32 bytes long.
* ``nonce`` (bytes): Initialization vector – typically 12 bytes for GCM.
* ``plaintext`` (bytes): Data to encrypt.
* ``associated_data`` (bytes or ``None``): Optional additional data that is
  authenticated but not encrypted. ``None`` is treated as an empty byte string.

It returns a dictionary containing:

* ``ciphertext`` (bytes): Encrypted data (without the authentication tag).
* ``tag`` (bytes): 16‑byte GCM authentication tag.

The ``cryptography`` library is used for the actual AES‑GCM operation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Accepted AES key sizes (in bytes) for AES‑128/192/256.
AES_KEY_SIZES = {16, 24, 32}

# GCM tag size is fixed at 16 bytes (128 bits) for the ``cryptography``
# implementation.
GCM_TAG_SIZE = 16


def solve(problem: Dict[str, Any]) -> Dict[str, bytes]:
    """Encrypt ``plaintext`` using AES‑GCM.

    The function mirrors the behaviour described in the reference
    implementation. It validates the key size, performs the encryption,
    separates the authentication tag from the ciphertext, and returns both
    components.

    Parameters
    ----------
    problem:
        Dictionary with ``key``, ``nonce``, ``plaintext`` and optionally
        ``associated_data``.

    Returns
    -------
    dict
        ``{"ciphertext": <bytes>, "tag": <bytes>}``
    """

    # Extract required fields; ``associated_data`` may be omitted or ``None``.
    key: bytes = problem["key"]
    nonce: bytes = problem["nonce"]
    plaintext: bytes = problem["plaintext"]
    associated_data = problem.get("associated_data") or b""

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    if len(key) not in AES_KEY_SIZES:
        raise ValueError(
            f"Invalid AES key size {len(key)}. Expected one of {sorted(AES_KEY_SIZES)}."
        )

    # ---------------------------------------------------------------------
    # Encryption using the cryptography library's high‑level AESGCM API.
    # ---------------------------------------------------------------------
    try:
        aesgcm = AESGCM(key)
        # ``encrypt`` returns ciphertext || tag.
        combined: bytes = aesgcm.encrypt(nonce, plaintext, associated_data)
    except Exception as exc:  # pragma: no cover – defensive programming
        logging.error("AES‑GCM encryption failed: %s", exc)
        raise

    # Ensure the combined output is long enough to contain a tag.
    if len(combined) < GCM_TAG_SIZE:
        raise ValueError(
            f"Encrypted output ({len(combined)} bytes) is shorter than the expected GCM tag size ({GCM_TAG_SIZE} bytes)."
        )

    # Separate the tag from the ciphertext.
    ciphertext: bytes = combined[:-GCM_TAG_SIZE]
    tag: bytes = combined[-GCM_TAG_SIZE:]

    return {"ciphertext": ciphertext, "tag": tag}
