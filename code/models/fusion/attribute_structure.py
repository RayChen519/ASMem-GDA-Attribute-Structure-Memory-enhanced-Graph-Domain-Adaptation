import torch
from torch import nn
from contracts.models.attribute_structure import AttributeStructureOutput
from models.attribute.branch import AttributeBranch
from models.structure.bga import StructureBranch


class AttributeStructure(nn.Module):
    def __init__(self):
        super().__init__()
        self.attribute = AttributeBranch()
        self.structure = StructureBranch()
        self.fusion = nn.Linear(256, 128)

    def forward(self, z, partition, *, source):
        h_a, a = self.attribute(z, source=source)
        h_lg, h_s, s = self.structure(z, partition, source=source)
        return AttributeStructureOutput(h_a, h_lg, h_s, self.fusion(torch.cat((h_a, h_s), 1)), a, s)
