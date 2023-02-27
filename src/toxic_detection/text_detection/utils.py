import unicodedata


def convert_to_unicode(text: str) -> str:
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""

    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def _is_punctuation(char: str) -> bool:
    """Checks whether `char` is a punctuation character.

    Treats all non-letter/number ASCII as punctuation.
    Characters such as "^", "$", and "`" are not in the Unicode
    Punctuation class but we treat them as punctuation anyways, for consistency.
    """
    cp = ord(char)
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_whitespace(char: str) -> bool:
    """Checks whether `char` is a whitespace character."""

    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_whitespace_or_punctuation(char: str) -> bool:
    """Checks whether `char` is a punctuation character
    or a whitespace character.

    Treats all non-letter/number ASCII as punctuation.
    Characters such as "^", "$", and "`" are not in the Unicode
    Punctuation class but we treat them as punctuation anyways, for consistency.
    """
    return _is_whitespace(char) or _is_punctuation(char)
