from torch import nn
from models.selection.operators import Selection


class AttributeBranch(nn.Module):
    attention_scope = 'full_domain_full_batch'

    def __init__(self):
        super().__init__()
        self.selection = Selection()

    def forward(self, z, *, source):
        return self.selection(z) if source else (z, None)
