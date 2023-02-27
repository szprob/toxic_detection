import math

import torch


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5*x*(1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421356))


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """Implementation of the gelu activation function
    currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def get_pad_mask(x: torch.Tensor) -> torch.Tensor:
    """The pad mask of x."""
    mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
    return mask
