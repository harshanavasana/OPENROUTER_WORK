import re

# Naive PII patterns: Email, SSN, Credit Cards (basic formats)
_EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_PATTERN = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")

def detect_pii(text: str) -> bool:
    """
    Detects simple PII to trigger edge-local privacy routing.
    If this returns True, the router will force the execution to use 
    a local on-device model rather than a cloud model.
    """
    if _EMAIL_PATTERN.search(text):
        return True
    if _SSN_PATTERN.search(text):
        return True
    if _CC_PATTERN.search(text):
        return True
    return False
