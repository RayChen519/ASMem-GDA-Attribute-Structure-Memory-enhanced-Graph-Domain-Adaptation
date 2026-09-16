from torch import nn
from models.selection.operators import Selection


class AttributeBranch(nn.Module):
    attention_scope = 'full_domain_full_batch'

    def __init__(self, options=None):
        super().__init__()
        options = options or {}
        self.symmetric = options.get("symmetric_selection", False)
        self.selection = Selection(options.get("scorer", "attention"), options.get("soft_threshold", True))

    def forward(self, z, *, source):
        return self.selection(z) if source or self.symmetric else (z, None)
