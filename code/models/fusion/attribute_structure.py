import torch
from torch import nn
from contracts.models.attribute_structure import AttributeStructureOutput
from models.attribute.branch import AttributeBranch
from models.structure.bga import StructureBranch

class AttributeStructure(nn.Module):
    def __init__(self, options=None):
        super().__init__()
        options = options or {}
        branches = options.get("branches", "both")
        self.attribute = AttributeBranch(options) if branches in ("both", "attribute") else None
        self.structure = StructureBranch(options) if branches in ("both", "structure") else None
        if options.get("shared_theta"):
            self.structure.selection.rho = self.attribute.selection.rho
        self.fusion = (nn.Linear(256,128) if branches == "both" else
                       nn.Sequential(nn.Linear(128,128), nn.LayerNorm(128)))

    def forward(self, z, partition, *, source):
        h_a, a = self.attribute(z, source=source) if self.attribute is not None else (z, None)
        h_lg, h_s, s = self.structure(z, partition, source=source) if self.structure is not None else (z,z,None)
        inputs = torch.cat((h_a,h_s),1) if self.attribute is not None and self.structure is not None else h_s if self.structure is not None else h_a
        return AttributeStructureOutput(h_a,h_lg,h_s,self.fusion(inputs),a,s)
