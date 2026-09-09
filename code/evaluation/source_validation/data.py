"""Validation labels are available only through the validation-side entry point."""
from contracts.data import SourceValidationView
from data.views.training import load_manifest, split_record, verified_tensor


def load_source_validation(manifest_path, source, label_rate, seed):
    root, manifest = load_manifest(manifest_path)
    split = split_record(manifest, source, label_rate, seed)
    labels = verified_tensor(root, split['validation_labels'])
    return SourceValidationView(labels['node_id'], labels['labels'], split['split_hash'])
