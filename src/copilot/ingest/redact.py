"""PII redaction with Presidio. Person names and usernames are deliberately kept (see design §6.6)."""

import ipaddress
import logging
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

REDACTED_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "SECRET",
)

_SECRET_PATTERNS = [
    Pattern("github_token", r"\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{30,}\b", 0.9),
    Pattern("github_pat", r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", 0.9),
    Pattern("openai_key", r"\bsk-[A-Za-z0-9_-]{20,}", 0.9),
    Pattern("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b", 0.9),
    Pattern("slack_token", r"\bxox[abposr]-[A-Za-z0-9-]{10,}", 0.9),
    Pattern("bearer_token", r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*", 0.9),
    Pattern("jwt", r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}", 0.9),
    Pattern("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", 0.95),
    # key=value / key: value assignments, e.g. password=..., API_KEY: '...'
    Pattern(
        "credential_assignment",
        r"(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\b\s*[=:]\s*['\"]?[^\s'\"]{6,}['\"]?",
        0.85,
    ),
]

# Loopback, bind-all, link-local, documentation, and reserved addresses are everyday dev-docs
# content ("run on 127.0.0.1:8000", "2001:db8::1", slice syntax like "[::2]"), not PII.
# Private ranges (10.x, 192.168.x) stay redacted: in a company repo they reveal internal infrastructure.
_DOC_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")]

# spaCy refuses texts over 1,000,000 characters, so long documents are redacted in slices.
_SLICE_CHARS = 100_000


@lru_cache
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    # Presidio warns about every non-English recognizer it skips; that's expected here.
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    nlp =NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            # We only use pattern-based entities; drop spaCy's NER labels to keep logs quiet.
            "ner_model_configuration": {
                "labels_to_ignore": ["CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "DATE", "TIME",
                                     "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT",
                                     "WORK_OF_ART", "LAW", "LANGUAGE"],
            },
        }
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    analyzer.registry.add_recognizer(PatternRecognizer(supported_entity="SECRET", patterns=_SECRET_PATTERNS))
    return analyzer, AnonymizerEngine()


def redact(text: str) -> str:
    if not text.strip():
        return text
    return "".join(_redact_slice(s) for s in _slices(text, _SLICE_CHARS))


def _is_safe_ip(match: str) -> bool:
    candidate = match.strip("[]").split("%")[0]  # drop brackets and IPv6 zone ids like %eth0
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False  # can't classify it: stay conservative and redact
    return (
        ip.is_loopback or ip.is_unspecified or ip.is_link_local or ip.is_reserved
        or any(ip in net for net in _DOC_NETS if net.version == ip.version)
    )


def _slices(text: str, limit: int) -> list[str]:
    """Split at newlines (else spaces) so no token, email, or key is cut in half."""
    out, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut <= start:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut + 1
        out.append(text[start:end])
        start = end
    return out


def _redact_slice(text: str) -> str:
    if not text.strip():
        return text
    analyzer, anonymizer = _engines()
    results = [
        r for r in analyzer.analyze(text=text, entities=list(REDACTED_ENTITIES), language="en")
        if not (r.entity_type == "IP_ADDRESS" and _is_safe_ip(text[r.start : r.end]))
    ]
    if not results:
        return text
    operators = {e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in REDACTED_ENTITIES}
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
