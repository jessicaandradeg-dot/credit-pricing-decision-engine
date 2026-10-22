import pytest
from cryptography.exceptions import InvalidTag

from credit_decision.security_demo import (
    decrypt_payload,
    encrypt_payload,
    generate_rsa_key_pair,
)


def test_hybrid_encryption_round_trip_uses_synthetic_bytes():
    private_key, public_key = generate_rsa_key_pair()
    payload = b'{"application_id":"SYN-CREDIT-1"}'

    envelope = encrypt_payload(payload, public_key)

    assert envelope.ciphertext != payload
    assert decrypt_payload(envelope, private_key) == payload


def test_authenticated_encryption_rejects_tampered_ciphertext():
    private_key, public_key = generate_rsa_key_pair()
    envelope = encrypt_payload(b"fictional", public_key)
    tampered = type(envelope)(
        encrypted_key=envelope.encrypted_key,
        nonce=envelope.nonce,
        ciphertext=envelope.ciphertext[:-1]
        + bytes([envelope.ciphertext[-1] ^ 1]),
    )

    with pytest.raises(InvalidTag):
        decrypt_payload(tampered, private_key)


def test_encryption_rejects_non_bytes_payload():
    _, public_key = generate_rsa_key_pair()

    with pytest.raises(TypeError, match="must be bytes"):
        encrypt_payload("not bytes", public_key)
