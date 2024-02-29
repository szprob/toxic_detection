import unicodedata
from typing import List, Optional, Set


def convert_to_unicode(text: str) -> str:
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""

    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def _is_control(char: str) -> bool:
    """Checks whether `chars` is a control character."""

    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
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


def _is_chinese_char(char: str):
    """Checks whether CP is the codepoint of a CJK character."""
    cp = ord(char)
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True
    return False


def whitespace_tokenize(text: str) -> List[str]:
    """Runs basic whitespace cleaning and splitting on a piece of text."""

    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def clean_text(text: str) -> str:
    """Performs invalid character removal and whitespace cleanup on text."""

    output = []
    for char in text:
        cp = ord(char)
        # valid 0xfffd 65533 replace \t \r \n
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def keep_ascii(text: str) -> str:
    """Only keep ascii chars"""
    filtered = (char for char in text if ord(char) <= 128)
    return "".join(filtered)


def run_split_on_punc(text: str, never_split: Optional[Set[str]] = None) -> List[str]:
    """Splits punctuation on a piece of text."""

    if never_split is not None and text in never_split:
        return [text]
    start_new_word = True
    output = []
    for char in text:
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)

    return ["".join(x) for x in output]


def tokenize_chinese_chars(text: str) -> str:
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        if _is_chinese_char(char):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def pad_list(text: List[int], maxlen: int = 25, pad: int = 0) -> List[int]:
    """Pads a int list."""

    x = [pad] * maxlen
    text_len = len(text)
    if text_len > 0:
        if text_len <= maxlen:
            x[:text_len] = text
        else:
            x = text[:maxlen]
    return x
