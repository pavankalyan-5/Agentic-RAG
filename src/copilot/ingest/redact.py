"""PII redaction with Presidio. Person names and usernames are deliberately kept (see design §6.6)."""

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
]


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
    analyzer, anonymizer = _engines()
    results = analyzer.analyze(text=text, entities=list(REDACTED_ENTITIES), language="en")
    if not results:
        return text
    operators = {e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in REDACTED_ENTITIES}
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
