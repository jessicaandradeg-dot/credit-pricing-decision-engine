# Cryptography learning demo

Run the demonstration from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m credit_decision.crypto_demo
python -m pytest tests/test_security_demo.py
```

The example generates an ephemeral 2048-bit RSA key pair, encrypts a fixed fictional byte string with AES-256-GCM, and wraps the one-time AES key with RSA-OAEP using SHA-256. AES-GCM authenticates the encrypted payload; tampering causes decryption to fail. A separate RSA-PSS/SHA-256 helper signs and verifies byte messages for the synthetic inter-agent demonstration. The private key exists only in process memory, and the command prints lengths and a round-trip status rather than plaintext.

This illustrates hybrid encryption and digital signatures as distinct primitives: symmetric cryptography handles application data, asymmetric cryptography protects the randomly generated symmetric key, and RSA-PSS authenticates signed bytes. A signature does not encrypt data or establish that its contents are true. These examples are not a key-management system, a protocol, or production-ready data protection. Do not use real personal or financial data, save the demo keys, or infer that the rest of this repository is secured by this example.
