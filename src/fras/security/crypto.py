"""Biometric template encryption at rest using Fernet (AES-128-CBC + HMAC-SHA256)."""

import os
from typing import Final

import numpy as np
from cryptography.fernet import Fernet

# 32-byte url-safe base64 key used in local testing/dev if env variable is unset
DEFAULT_DEV_KEY: Final[bytes] = b"vJJ0VJWoQLkTewC8A6QYrecdG-ttmESz0JVdMGr2V2I="


def get_cipher(key: str | bytes | None = None) -> Fernet:
    """Returns a Fernet cipher instance initialized from environment or provided key."""
    if key is not None:
        raw_key = key.encode("utf-8") if isinstance(key, str) else key
        return Fernet(raw_key)

    env_key = os.getenv("BIOMETRIC_ENCRYPTION_KEY")
    if env_key:
        return Fernet(env_key.encode("utf-8"))

    return Fernet(DEFAULT_DEV_KEY)


def encrypt_embedding(embedding: np.ndarray, key: str | bytes | None = None) -> str:
    """Serializes and encrypts an embedding float vector into a Fernet ciphertext token.

    Args:
        embedding: Numpy float vector (e.g. 128-d or 512-d).
        key: Optional custom Fernet key.

    Returns:
        Base64-encoded ciphertext string.
    """
    cipher = get_cipher(key)
    raw_bytes = embedding.astype(np.float32).tobytes()
    token = cipher.encrypt(raw_bytes)
    return token.decode("utf-8")


def decrypt_embedding(encrypted_str: str, key: str | bytes | None = None) -> np.ndarray:
    """Decrypts a Fernet ciphertext token and reconstructs the numpy float vector.

    Args:
        encrypted_str: Base64-encoded ciphertext string.
        key: Optional custom Fernet key.

    Returns:
        1D numpy float32 array.
    """
    cipher = get_cipher(key)
    decrypted_bytes = cipher.decrypt(encrypted_str.encode("utf-8"))
    return np.frombuffer(decrypted_bytes, dtype=np.float32).copy()
