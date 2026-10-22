"""Run local encryption and signature checks over fictional sample data."""

import json

from .security_demo import (
    decrypt_payload,
    encrypt_payload,
    generate_rsa_key_pair,
    sign_message,
    verify_message,
)


def main() -> None:
    private_key, public_key = generate_rsa_key_pair()
    sample = json.dumps(
        {"application_id": "SYN-CREDIT-DEMO", "amount_k": 12.5},
        separators=(",", ":"),
    ).encode("utf-8")
    envelope = encrypt_payload(sample, public_key)
    restored = decrypt_payload(envelope, private_key)
    signature = sign_message(sample, private_key)
    verify_message(sample, signature, public_key)

    print(
        json.dumps(
            {
                "purpose": "educational synthetic-data demo; not production security",
                "symmetric_encryption": "AES-256-GCM",
                "asymmetric_key_wrapping": "RSA-OAEP with SHA-256",
                "digital_signature": "RSA-PSS with SHA-256",
                "plaintext_bytes": len(sample),
                "ciphertext_bytes": len(envelope.ciphertext),
                "round_trip_verified": restored == sample,
                "signature_verified": True,
                "keys": "ephemeral; never saved",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
