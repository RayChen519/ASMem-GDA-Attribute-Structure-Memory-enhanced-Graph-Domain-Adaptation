from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    node_id: torch.Tensor
    degree: torch.Tensor
    domain_id: torch.Tensor
    feature_dim: int
    attribute_union_hash: str
    dataset_version: str
    gcn_edge_index: torch.Tensor
    gcn_edge_weight: torch.Tensor
    source_train_mask: torch.Tensor
    source_val_mask: torch.Tensor
    source_unlabeled_mask: torch.Tensor
    target_unlabeled_mask: torch.Tensor
    target_test_mask: torch.Tensor

    @property
    def normalized_adjacency(self):
        n = self.x.shape[0]
        return torch.sparse_coo_tensor(self.gcn_edge_index, self.gcn_edge_weight,
                                       (n, n), check_invariants=True).coalesce()


@dataclass(frozen=True, slots=True)
class SourceTrainView:
    graph: GraphData
    train_node_id: torch.Tensor
    train_y: torch.Tensor
    split_hash: str


@dataclass(frozen=True, slots=True)
class SourceValidationView:
    node_id: torch.Tensor
    validation_y: torch.Tensor
    split_hash: str


@dataclass(frozen=True, slots=True)
class TargetTrainView:
    x: torch.Tensor
    edge_index: torch.Tensor
    node_id: torch.Tensor
    target_unlabeled_mask: torch.Tensor
    domain_id: torch.Tensor
    degree: torch.Tensor
    feature_dim: int
    attribute_union_hash: str
    dataset_version: str
    gcn_edge_index: torch.Tensor
    gcn_edge_weight: torch.Tensor

    @property
    def normalized_adjacency(self):
        n = self.x.shape[0]
        return torch.sparse_coo_tensor(self.gcn_edge_index, self.gcn_edge_weight,
                                       (n, n), check_invariants=True).coalesce()
