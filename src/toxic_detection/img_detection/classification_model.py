from typing import Dict, List, Optional

import torch
import torchvision
from torch import nn


class Classifier(nn.Module):
    """Toxic Classifier.

    Given a transformed image,`classifier` will get a toxic socre on it.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__()
        self.config = {} if config is None else config

        self.resnet = torchvision.models.resnet50()
        self.resnet.fc = nn.Linear(
            in_features=self.config.get("in_features", 2048),
            out_features=self.config.get("tag_num", 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resnet(x)
        return out

    @torch.no_grad()
    def score(self, input: torch.Tensor) -> List[float]:
        """Scoring the input image(one input).

        Args:
            input (torch.Tensor):
                img input(should be transformed).

        Returns:
            List[float]:
                The toxic score of the input .
        """

        return (
            torch.softmax(self.forward(input), dim=1).detach().cpu().view(-1).tolist()
        )
