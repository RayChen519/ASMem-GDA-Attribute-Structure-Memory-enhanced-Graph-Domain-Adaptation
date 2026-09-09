from typing import NamedTuple
import torch


class AttributeStructureOutput(NamedTuple):
    """H^A, H^LG, H^S, H^AS: [N_d,128], original node_id row order.

    Future Memory consumes h_s as Query/Key and h_as as Value/residual.
    Thresholded tensors are the pre-multiplication Source sparsity operands;
    Target has None for both because its selection operators are not called.
    """
    h_a: torch.Tensor
    h_lg: torch.Tensor
    h_s: torch.Tensor
    h_as: torch.Tensor
    attribute_thresholded: torch.Tensor | None
    structure_thresholded: torch.Tensor | None
