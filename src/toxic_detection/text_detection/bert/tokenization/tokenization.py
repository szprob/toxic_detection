# coding=utf-8
import collections
from typing import Dict, Iterable, List, Optional, Union

import torch

from toxic_detection.module_utils import PreTrainedModule
from .basic_tokenization import BasicTokenizer
from .bpe import BPE
from .tokenization_utils import pad_list


class Tokenizer(PreTrainedModule):
    """Tokenization classes modified from berttokenizer.

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
        max_token_length (int, optional):
            Model will not tokenize words longer than `max_token_length`.
            It will return `unk_token` instead.
            Defaults to 100.
        maxlen (int, optional):
            Model will tokenize sequence no longer than `maxlen`.
            Defaults to 512.
        unk_token (str, optional):
            A token that is not in the vocabulary cannot be converted to an ID
            and is set to be this token instead
            Defaults to "[UNK]".
        vocab (Optional[Union[Dict,Set]], optional):
            Vocab for tokenization.
            Defaults to None.
        sep_token (str, optional):
            The separator token.
            which is used when building a sequence from multiple sequences.
            Defaults to "[SEP]".
        pad_token (str, optional):
            The token used for padding.
            for example when batching sequences of different lengths.
            Defaults to "[PAD]".
        cls_token (str, optional):
            The classifier token which is used when doing sequence classification.
            It is the first token of the sequence when built with special tokens.
            Defaults to "[CLS]".
        mask_token (str, optional):
            The token used for masking values.
            This is the token used when training this model with mlm.
            This is the token which the model will try to predict.
            Defaults to "[MASK]".

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
        max_token_length: int = 100,
        vocab: Optional[Dict[str, int]] = None,
        maxlen: int = 512,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
    ):
        super().__init__()

        # basic tokens
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.unk_token = unk_token

        self.maxlen = maxlen

        # basic set
        self.do_lower_case = do_lower_case
        self.strip_accents = strip_accents
        self.tokenize_chinese = tokenize_chinese
        self.only_ascii = only_ascii
        self.split_on_punc = split_on_punc

        # never split
        if never_split is None:
            never_split = set()
        self.never_split = set(never_split)

        # BasicTokenizer
        self._basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            strip_accents=strip_accents,
            tokenize_chinese=tokenize_chinese,
            only_ascii=only_ascii,
            split_on_punc=split_on_punc,
            never_split=never_split,
        )

        # bpe set
        if vocab is None:
            vocab = {}
        self.vocab = vocab
        self.max_token_length = max_token_length

        self._wordpiece_tokenizer = BPE(
            vocab=vocab,
            unk_token=unk_token,
            max_token_length=max_token_length,
        )

    def load(self, model: Union[str, Dict]) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (Union[str, Dict]):
                Model file need to be loaded.
                Can be either:
                    - A string, the path of a pretrained model.
                    - A state dict containing model weights.

        Raises:
            ValueError: model file should be a dict.
        """

        if isinstance(model, str):
            model_file = self._load_pkl(model)
        else:
            model_file = model

        if not isinstance(model_file, Dict):
            raise ValueError("""model file should be a dict!""")

        self.set_params(model_file)
        self.init_model()

    def set_params(self, model_file: Dict) -> None:
        """Set model params from `model_file`.

        Args:
            model_file (Dict):
                Dict containing model params.
        """

        if "vocab" in model_file:
            if isinstance(model_file["vocab"], Dict):
                self.vocab = model_file["vocab"]

    def init_model(self) -> None:
        """Init tokenizer after loading params.

        You should init model before using the tokenizer.
        Usually, we can run this function after loading pretrained model.
        Running this code manually is not recommended.

        Init function including :
            1. building bpe
            2. get token id

        """

        self._wordpiece_tokenizer = BPE(
            vocab=self.vocab,
            unk_token=self.unk_token,
            max_token_length=self.max_token_length,
        )

        if self.sep_token not in self.vocab:
            raise ValueError("""vocab should include sep_token!""")
        self.sep_token_id = self.vocab.get(self.sep_token)

        if self.pad_token not in self.vocab:
            raise ValueError("""vocab should include pad_token!""")
        self.pad_token_id = self.vocab.get(self.pad_token)

        if self.cls_token not in self.vocab:
            raise ValueError("""vocab should include cls_token!""")
        self.cls_token_id = self.vocab.get(self.cls_token)

        if self.mask_token not in self.vocab:
            raise ValueError("""vocab should include mask_token!""")
        self.mask_token_id = self.vocab.get(self.mask_token)

        if self.unk_token not in self.vocab:
            raise ValueError("""vocab should include unk_token!""")
        self.unk_token_id = self.vocab.get(self.unk_token)

        # id2token
        self._ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )

    def add_never_split(self, never_split: Iterable[str]) -> None:
        """Adds additional nerver split words to tokenizer."""
        self.never_split.update(set(never_split))
        self._basic_tokenizer.add_never_split(never_split)

    def tokenize(self, text: str) -> List[str]:
        """Run bert tokenization on `text`."""

        tokens = self._basic_tokenizer.tokenize(text)
        split_tokens = []
        for token in tokens:
            if token in self.never_split:
                split_tokens.append(token)
            else:
                split_tokens.extend(self._wordpiece_tokenizer.tokenize(token))
        return split_tokens

    def convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_tokens_to_id(self, tokens: Iterable) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens).replace(" ##", "").strip()

    def encode(self, text: str, maxlen: Optional[int] = None) -> List[int]:
        """Encode text to list of int,result will be padded with zeros."""
        if maxlen is None:
            maxlen = self.maxlen
        tokens = self.tokenize(text)
        tokens = tokens[: maxlen - 2]
        ids = self.convert_tokens_to_id(tokens)
        ids = [self.cls_token_id] + ids + [self.sep_token_id]
        ids = pad_list(ids, maxlen, self.pad_token_id)
        return ids

    def encode_seqs(
        self, seqs: Iterable[str], maxlen: Optional[int] = None
    ) -> List[int]:
        """Encode seqs to list of int,result will be padded with zeros."""
        if maxlen is None:
            maxlen = self.maxlen
        seq_tokens = [self.tokenize(text) for text in seqs]
        seq_ids = [self.convert_tokens_to_id(tokens) for tokens in seq_tokens]
        out = [self.cls_token]
        for ids in seq_ids:
            ids = ids[: maxlen - 1 - len(out)]
            out = out + ids + [self.sep_token_id]
            if len(out) >= maxlen:
                break
        out = pad_list(out, maxlen, self.pad_token_id)
        return out

    def encode_tensor(self, text: str, maxlen: Optional[int] = None) -> torch.Tensor:
        """Encode text to long tensor,result will be padded with zeros."""
        return torch.tensor(self.encode(text, maxlen)).long()

    def encode_seqs_tensor(
        self, seqs: Iterable[str], maxlen: Optional[int] = None
    ) -> torch.Tensor:
        """Encode seqs to long tensor,result will be padded with zeros."""
        return torch.tensor(self.encode_seqs(seqs, maxlen)).long()
