from typing import NamedTuple

import torch


class EncoderOutput(NamedTuple):
    """H1 is post-dropout; Z is post-LayerNorm with no final ReLU.

    Both tensors preserve the input node order. A/S consumes Z directly.
    """
    h1: torch.Tensor
    z: torch.Tensor
