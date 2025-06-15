import re, hashlib
from typing import Tuple

def redact_pii(text: str) -> str:
    """Redact simple PII patterns (email, phone, street numbers)."""
    if not text:
        return text
    # Email
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL]", text)
    # Phone numbers (very naive) – sequences of 7+ digits or digit separators
    text = re.sub(r"\+?\d[\d\s\-]{6,}\d", "[PHONE]", text)
    # Street numbers (123 Main St.)
    text = re.sub(r"\d+\s+\w+\s+(?:St\.|Street|Ave\.|Avenue|Rd\.|Road)", "[ADDRESS]", text, flags=re.I)
    return text


def hash_prompt(prompt: str) -> str:
    """Return SHA-256 hex of a prompt for logging without storing full text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest() 