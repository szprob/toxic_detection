from typing import Optional

import torch
from torch import nn

from toxic_detection.text_detection.bert.attention import MultiHeadAttention
from toxic_detection.text_detection.bert.pff import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Encoder Layer for bert.

    Attributes:
        d_model (int, optional):
            The dim of hidden layer.
            Defaults to 512.
        num_heads (int, optional):
            The number of attention heads.
            Defaults to 8.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = PositionwiseFeedForward(d_model=d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x (b , l ,d )
        # enc_inputs to same Q,K,V
        x = self.attn(x, x, x, attn_mask)
        x = self.ffn(x)
        # x [b x len_q x d_model]
        return x


class Encoder(nn.Module):
    """Encoder for bert.

    Attributes:
        d_model (int, optional):
            The dim of hidden layer.
            Defaults to 512.
        num_heads (int, optional):
            The number of attention heads.
            Defaults to 8.
        n_layers (int, optional):
            The number of encoder layers.
            Defaults to 8.

    """

    def __init__(
        self, d_model: int = 512, num_heads: int = 8, n_layers: int = 8
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(d_model=self.d_model, num_heads=self.num_heads)
            for layer in range(self.n_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x(b , l ,d )
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x, attn_mask)
        return x
