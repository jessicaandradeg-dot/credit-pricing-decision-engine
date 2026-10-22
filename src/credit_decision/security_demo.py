"""Educational hybrid-encryption example for synthetic payloads only."""

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ASSOCIATED_DATA = b"synthetic-credit-demo:v1"


@dataclass(frozen=True)
class EncryptedEnvelope:
    """Encrypted bytes and the wrapped one-time AES key."""

    encrypted_key: bytes
    nonce: bytes
    ciphertext: bytes


def generate_rsa_key_pair() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """Create an ephemeral RSA key pair for the local educational demo."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key, private_key.public_key()


def encrypt_payload(
    plaintext: bytes,
    recipient_public_key: rsa.RSAPublicKey,
) -> EncryptedEnvelope:
    """Encrypt bytes with AES-256-GCM and wrap the random key with RSA-OAEP."""
    if not isinstance(plaintext, bytes):
        raise TypeError("plaintext must be bytes.")

    aes_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext, _ASSOCIATED_DATA)
    encrypted_key = recipient_public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return EncryptedEnvelope(
        encrypted_key=encrypted_key,
        nonce=nonce,
        ciphertext=ciphertext,
    )


def decrypt_payload(
    envelope: EncryptedEnvelope,
    recipient_private_key: rsa.RSAPrivateKey,
) -> bytes:
    """Unwrap the AES key and authenticate/decrypt the payload."""
    aes_key = recipient_private_key.decrypt(
        envelope.encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return AESGCM(aes_key).decrypt(
        envelope.nonce,
        envelope.ciphertext,
        _ASSOCIATED_DATA,
    )
