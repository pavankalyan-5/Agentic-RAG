import pytest

from copilot.ingest.redact import redact


@pytest.mark.parametrize(
    "raw,token",
    [
        ("mail me at jane.doe@example.com", "<EMAIL_ADDRESS>"),
        # Not 078-05-1120: that famous "Woolworth" SSN is on Presidio's known-invalid list.
        ("SSN 536-22-8726 here", "<US_SSN>"),
        ("card 4111 1111 1111 1111", "<CREDIT_CARD>"),
        ("call +1 415 555 2671", "<PHONE_NUMBER>"),
        ("server at 10.20.30.40", "<IP_ADDRESS>"),
        ("token ghp_" + "a" * 36, "<SECRET>"),
        ("key sk-proj-" + "B" * 40, "<SECRET>"),
        ("aws AKIA" + "Q" * 16, "<SECRET>"),
        # Review #7: formats that used to pass through untouched.
        ("Authorization: Bearer " + "abcDEF123" * 4, "<SECRET>"),
        ("slack xoxb-1234567890-abcdefABCDEF", "<SECRET>"),
        ("password=hunter2hunter2", "<SECRET>"),
        ("API_KEY: 'zx81zx81zx81'", "<SECRET>"),
        ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlX2hlcmU", "<SECRET>"),
    ],
)
def test_redacts(raw, token):
    assert token in redact(raw)


@pytest.mark.parametrize(
    "safe",
    [
        "see #4521 and v0.110.0", "thanks @tiangolo", "Sebastián Ramírez fixed it", "",
        # Review #8: loopback and bind-all addresses are core dev-docs content, not PII.
        "run uvicorn on 127.0.0.1:8000", "bind to 0.0.0.0 in Docker", "use the bearer token from settings",
        # Found on real Starlette PRs: IPv6 loopback/link-local/documentation addresses and slice syntax.
        "Host: [::1]:8080", "zone ids like fe80::1%eth0", "example 2001:db8::1 per RFC 3849",
        "memoryview(data)[::2]", "doc example 192.0.2.10",
    ],
)
def test_leaves_non_pii_untouched(safe):
    assert redact(safe) == safe


@pytest.mark.parametrize("raw", ["db at 192.168.1.20", "vpn 10.0.0.5", "server 8.8.4.4", "node fd12:3456::7"])
def test_private_and_public_ips_still_redacted(raw):
    assert "<IP_ADDRESS>" in redact(raw)


def test_private_key_block_fully_redacted():
    key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow" + "A" * 200 + "\n-----END RSA PRIVATE KEY-----"
    out = redact(f"oops:\n{key}\nthanks")
    assert "MIIEow" not in out and "<SECRET>" in out and out.endswith("thanks")


def test_very_long_text_is_redacted_without_crashing():
    # Review #2: spaCy refuses texts over 1,000,000 characters, which aborted the whole sync.
    text = "lorem ipsum " * 100_000 + "mail jane.doe@example.com" + " dolor" * 10
    out = redact(text)
    assert "jane.doe@example.com" not in out and "<EMAIL_ADDRESS>" in out
    assert out.startswith("lorem ipsum ")
