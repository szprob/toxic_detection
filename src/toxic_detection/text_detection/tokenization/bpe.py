from typing import Dict, Iterable, List, Optional, Set, Union

from toxic_detection.text_detection.tokenization.tokenization_utils import (
    whitespace_tokenize,
)


class BPE:
    """A implementation of wordpiece algorithm modified from berttokenizer.

    Attributes:
        max_token_length (int, optional):
            Model will not tokenize words longer than `max_token_length`.
            It will return `unk_token` instead.
            Defaults to 100.
        unk_token (str, optional):
            Unknown words token.
            Defaults to "[UNK]".
        vocab (Optional[Union[Dict,Set]], optional):
            Vocab for tokenization.
            Defaults to None.

    """

    def __init__(
        self,
        *,
        max_token_length: int = 100,
        unk_token: str = "[UNK]",
        vocab: Optional[Union[Dict, Set]] = None,
    ):
        self.max_token_length = max_token_length
        self.unk_token = unk_token
        self.vocab = vocab

    def tokenize(self, text: Union[str, Iterable[str]]) -> List[str]:
        """Tokenization of a piece of text.

        Args:
            text (Union[str,Iterable[str]]):
                Text for bpe tokenization.

        Returns:
            List[str]:
                Text after bpe tokenization.
        """

        output_tokens = []
        if isinstance(text, str):
            text = whitespace_tokenize(text)
        for token in text:
            chars = list(token)
            if len(chars) > self.max_token_length:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
