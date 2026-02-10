import re

MIN_TOKENS = 2

def split_clauses(text: str) -> list[str]:
    """
    Découpe une phrase en queries
    """
    text = text.lower()
    separators = r"\b(et|ou|mais|car|parce que|donc|alors|puis|ensuite|donc|or|car|cependant)\b|[.,;!?]"
    parts = re.split(separators, text, flags=re.IGNORECASE)

    queries = [
        p.strip()
        for p in parts
        if p and len(p.split()) >= MIN_TOKENS
    ]

    return queries
