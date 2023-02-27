from typing import Dict, Optional

import torch
from torch import nn

from toxic_detection.text_detection.bert.bert import BERT


class Classifier(nn.Module):
    """Toxic Classifier.

    Given a encoded text,`classifier` will get a toxic socre on it.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = {} if config is None else config
        self.bert = BERT(config=self.config)
        self.cls_head = nn.Sequential(
            nn.Linear(
                self.config.get("hidden_size", 512), self.config.get("hidden_size", 512)
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(
                self.config.get("hidden_size", 512), self.config.get("tag_num", 6)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqs = self.bert(x)
        out = torch.mean(seqs, dim=1)
        out = self.cls_head(out)
        return out

    @torch.no_grad()
    def score(self, input: torch.Tensor) -> torch.Tensor:
        """Scoring the input text(one input).

        Args:
            input (torch.Tensor):
                Text input(should be encoded by bert tokenizer.)

        Returns:
            torch.Tensor:
                The toxic score of the input .
        """

        return torch.sigmoid(self.forward(input)).detach().cpu()
