from typing import Optional

import torch
from torch import nn


class Embeddings(nn.Module):
    """Embedding layer for bert.

    Attributes:
        vocab_size (int):
            Vocab size.
        hidden_size (int, optional):
            Hidden dim of hidden layer.
            Defaults to 512.
        maxlen (int, optional):
            Max length of sequence.
            Defaults to 512.
        segment_size (int, optional):
            Segment number of bert.
            This attribute is not usually used .
            Defaults to 1.

    """

    def __init__(
        self, vocab_size: int, hidden_size: int = 512, maxlen=512, segment_size=1
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(maxlen, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_size, hidden_size)
        self.layerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(
        self, token_ids: torch.Tensor, segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of embedding.

        Args:
            token_ids (torch.Tensor):
                The index of words. shape:(b,l)
            segment_ids (Optional[torch.Tensor], optional):
                The index of segments.
                This arg is not usually used.
                Defaults to None.

        Returns:
            torch.Tensor:
                Embedding result of token_ids. shape:(b,l,d)
        """

        seq_length = token_ids.size(1)

        # pos embed
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=token_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # word embed
        words_embeddings = self.word_embeddings(token_ids)

        # seg embed
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        # sum of embeddings
        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layerNorm(embeddings)

        return embeddings
