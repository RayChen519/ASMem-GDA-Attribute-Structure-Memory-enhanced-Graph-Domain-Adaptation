import torch
from torch.nn import functional as F


def classification_loss(classifier, source_output, train_ids, train_y):
    return F.cross_entropy(classifier(source_output.h_as[train_ids]), train_y)


def sparsity_loss(source_output):
    a, s = source_output.attribute_thresholded, source_output.structure_thresholded
    available = [x for x in (a,s) if x is not None]
    return sum((x.abs().mean() for x in available), source_output.h_as.new_zeros(()))



def domain_loss(discriminator, source_output, target_output):
    # Mean BCE across all Source/Target nodes; Source=0, Target=1.
    h = torch.cat((source_output.h_as, target_output.h_as))
    labels = torch.cat((h.new_zeros((len(source_output.h_as), 1)),
                        h.new_ones((len(target_output.h_as), 1))))
    return F.binary_cross_entropy_with_logits(discriminator(h), labels)
