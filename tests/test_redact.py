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
    ],
)
def test_redacts(raw, token):
    assert token in redact(raw)


@pytest.mark.parametrize(
    "safe",
    ["see #4521 and v0.110.0", "thanks @tiangolo", "Sebastián Ramírez fixed it", ""],
)
def test_leaves_non_pii_untouched(safe):
    assert redact(safe) == safe
