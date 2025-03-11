import re
import string
import unicodedata

def normalize_text(s):
    # Normalize Unicode characters
    s = unicodedata.normalize("NFD", s)
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles (a, an, the)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Fix extra whitespaces
    s = " ".join(s.split())
    return s