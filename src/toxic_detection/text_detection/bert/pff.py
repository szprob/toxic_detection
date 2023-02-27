from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    """PositionwiseFeedForward layer of bert.

    Attributes:
        d_model (int, optional):
            The dim of hidden layer.
            Defaults to 512.
        d_ff (Optional[int], optional):
            The dim of feed forward layer.
            If None, the value will be set to 4 * `d_model`.
            Defaults to None.

    """

    def __init__(self, d_model: int = 512, d_ff: Optional[int] = None) -> None:
        super(PositionwiseFeedForward, self).__init__()
        if not d_ff:
            d_ff = 4 * d_model
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.fc_2(F.gelu(self.fc_1(x)))
        x = self.norm(x + res)
        return x
