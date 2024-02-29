# coding=utf-8
from typing import Iterable, List, Optional

from .tokenization_utils import (
    clean_text,
    convert_to_unicode,
    keep_ascii,
    run_split_on_punc,
    run_strip_accents,
    tokenize_chinese_chars,
    whitespace_tokenize,
)


class BasicTokenizer:
    """Basic tokenization classes modified from berttokenizer.

    Attributes:
        do_lower_case (bool, optional):
            Whether or not to lowercase the input when tokenizing.
            Defaults to True.
        strip_accents (bool, optional):
            Whether or not to strip all accents.
            If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            Defaults to True.
        tokenize_chinese (bool, optional):
            Whether or not to tokenize Chinese characters.
            Defaults to True.
        only_ascii (bool, optional):
            Whether or not to keep only ascii characters.
            If True,`tokenize_chinese` will not work.
            Defaults to False.
        split_on_punc (bool, optional):
            Whether or not to split on punctuation.
            Defaults to True.
        never_split (Optional[Iterable[str]] , optional):
            Collection of tokens which will never be split during tokenization.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        do_lower_case: bool = True,
        strip_accents: bool = True,
        tokenize_chinese: bool = True,
        only_ascii: bool = False,
        split_on_punc: bool = True,
        never_split: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()

        self.do_lower_case = do_lower_case
        self.strip_accents = strip_accents
        self.tokenize_chinese = tokenize_chinese
        self.only_ascii = only_ascii
        self.split_on_punc = split_on_punc

        if never_split is None:
            never_split = set()
        self.never_split = set(never_split)

    def add_never_split(self, never_split: Iterable[str]) -> None:
        """Adds additional nerver split words to tokenizer."""
        never_split = set(never_split)
        length_before = len(self.never_split)
        self.never_split.update(never_split)
        length_after = len(self.never_split)
        print(f"Adds {length_after-length_before} unique words.")

    def tokenize(self, text: str) -> List[str]:
        """Run basic tokenization on text.

        Args:
            text (str):
                Text for tokenization.

        Returns:
            List[str]:
                Tokens after tokenization.
        """

        text = clean_text(convert_to_unicode(text))
        if self.tokenize_chinese:
            text = tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in self.never_split:
                if self.only_ascii:
                    token = keep_ascii(token)
                if self.do_lower_case:
                    token = token.lower()
                if self.strip_accents:
                    token = run_strip_accents(token)
            if self.split_on_punc:
                split_tokens.extend(run_split_on_punc(token, self.never_split))
            else:
                split_tokens.append(token)
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
