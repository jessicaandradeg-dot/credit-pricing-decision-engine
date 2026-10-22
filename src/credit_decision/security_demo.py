"""Educational hybrid-encryption example for synthetic payloads only."""

import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
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


def sign_message(message: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
    """Sign bytes with RSA-PSS/SHA-256 for the educational demo."""
    if not isinstance(message, bytes):
        raise TypeError("message must be bytes.")
    return private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )


def verify_message(
    message: bytes,
    signature: bytes,
    public_key: rsa.RSAPublicKey,
) -> None:
    """Verify RSA-PSS/SHA-256; invalid signatures raise InvalidSignature."""
    if not isinstance(message, bytes):
        raise TypeError("message must be bytes.")
    if not isinstance(signature, bytes):
        raise TypeError("signature must be bytes.")
    public_key.verify(
        signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )


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
