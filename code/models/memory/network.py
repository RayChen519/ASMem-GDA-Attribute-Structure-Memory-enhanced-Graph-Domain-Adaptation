"""Structural addressing with Source attribute/structure values (P4)."""
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class MemoryOutput(NamedTuple):
    logits: torch.Tensor
    probability: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    similarity: torch.Tensor
    attention: torch.Tensor
    read: torch.Tensor
    fused: torch.Tensor


class MemoryNetwork(nn.Module):
    def __init__(self, num_classes, temperature=.1):
        super().__init__()
        if num_classes < 2 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid Memory classes/temperature')
        self.temperature = temperature
        self.query = nn.Linear(128, 128, bias=False)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.fusion = nn.Linear(256, 128, bias=False)
        # P4's trainable whitelist has no LayerNorm affine parameters.
        self.norm = nn.LayerNorm(128, elementwise_affine=False)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, h_s, h_as, anchor_h_s, anchor_h_as, query_ids, anchor_ids, *, source):
        n, k = len(h_s), len(anchor_h_s)
        if k == 0 or any(t.shape != (size, 128) for t, size in
                         ((h_s, n), (h_as, n), (anchor_h_s, k), (anchor_h_as, k))):
            raise ValueError('Memory requires [N,128] queries and nonempty [K,128] anchors')
        for ids, size in ((query_ids, n), (anchor_ids, k)):
            if ids.dtype != torch.long or ids.shape != (size,) or ids.unique().numel() != size:
                raise ValueError('Invalid or duplicate Memory node IDs')
        q = F.normalize(self.query(h_s), dim=-1)
        key = F.normalize(self.key(anchor_h_s), dim=-1)
        value = self.value(anchor_h_as)
        e = q @ key.T / self.temperature
        if source:
            e = e.masked_fill(query_ids[:, None] == anchor_ids[None, :], -torch.inf)
        # A one-anchor query has no available neighbor: define a zero read.
        empty = torch.isneginf(e).all(-1, keepdim=True)
        a = e.masked_fill(empty, 0).softmax(-1).masked_fill(empty, 0)
        read = a @ value
        fused = self.norm(h_as + self.fusion(torch.cat((h_as, read), -1)))
        logits = self.classifier(fused)
        return MemoryOutput(logits, logits.softmax(-1), q, key, value, e, a, read, fused)
