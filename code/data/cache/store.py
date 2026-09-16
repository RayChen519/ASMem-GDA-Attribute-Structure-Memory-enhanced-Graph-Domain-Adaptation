import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import torch


def io_path(path):
    """Use extended Windows paths for hash-named artifacts in deep workspaces."""
    path = Path(path).resolve()
    value = str(path)
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = ('\\\\?\\UNC\\' + value[2:]) if value.startswith('\\\\') else ('\\\\?\\' + value)
    return Path(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with io_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _replace_with_retry(source, destination):
    """Bounded retries for Windows access/sharing violations; never delete destination."""
    delays = (0.05, 0.1, 0.2, 0.4, 0.8)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            if attempt == len(delays):
                error.add_note(f'Atomic replacement failed after {attempt + 1} attempts: '
                               f'{destination}. Check file attributes, ACL and open handles.')
                raise
            time.sleep(delays[attempt])


def atomic_write(path, writer):
    path = io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    os.close(fd)
    try:
        writer(Path(temporary))
        _replace_with_retry(temporary, path)
    except BaseException as error:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f'Temporary file cleanup also failed: {temporary}: {cleanup_error}')
        raise


def write_json(path, value):
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2,
                 ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8'))


def read_json(path):
    return json.loads(io_path(path).read_text(encoding='utf-8'))


def save_tensor(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic_write(path, lambda p: p.write_bytes(buffer.getvalue()))


FIELDS = {
    'features': {'dataset_version', 'domain', 'attribute_union_hash', 'normalization_version'},
    'split': {'dataset_version', 'source_domain', 'label_rate', 'split_seed'},
    'metis': {'dataset_version', 'domain', 'P', 'metis_seed'},
    'anchor': {'dataset_version', 'direction', 'label_rate', 'split_seed', 'K',
               'sampler', 'sampler_hparams'},
}


def cache_key(kind, **inputs):
    if kind not in FIELDS or not FIELDS[kind] <= inputs.keys():
        raise ValueError(f'Missing decisive inputs for {kind}')
    return digest({'kind': kind, 'inputs': inputs})


class Cache:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, kind, inputs):
        return self.root / kind / (cache_key(kind, **inputs) + '.pt')

    def get_or_create(self, kind, inputs, factory):
        path = self.path(kind, inputs)
        meta = path.with_suffix('.json')
        if io_path(path).exists() or io_path(meta).exists():
            record = read_json(meta)
            if record['inputs'] != inputs or record['sha256'] != file_hash(path):
                raise ValueError(f'Corrupt cache: {path}')
            return torch.load(io_path(path), weights_only=True), path, True
        value = factory()
        save_tensor(path, value)
        write_json(meta, {'kind': kind, 'inputs': inputs, 'sha256': file_hash(path)})
        return value, path, False
