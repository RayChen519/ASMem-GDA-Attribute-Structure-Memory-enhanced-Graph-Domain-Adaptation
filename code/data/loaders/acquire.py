"""Acquire only pinned MAT files; never reuse upstream processed graph caches."""
import argparse
import shutil
import urllib.request
from pathlib import Path

from data.cache.store import atomic_write, file_hash, read_json, write_json


def acquire(code_root, local_dir=None):
    root = Path(code_root)
    release = read_json(root / 'configs/base/dataset_release.json')
    destination = root / 'artifacts/datasets/raw'
    for domain, entry in release['domains'].items():
        output = destination / domain / entry['filename']
        url = f"https://raw.githubusercontent.com/joe817/SGDA/{release['commit']}/data/{entry['filename']}"
        if not output.exists():
            def fetch(temporary):
                if local_dir is not None:
                    shutil.copyfile(Path(local_dir) / entry['filename'], temporary)
                else:
                    with urllib.request.urlopen(url, timeout=90) as source, temporary.open('wb') as target:
                        shutil.copyfileobj(source, target)
                if file_hash(temporary) != entry['sha256']:
                    raise ValueError(f'{domain}: raw release checksum mismatch')
            atomic_write(output, fetch)
        if file_hash(output) != entry['sha256']:
            raise ValueError(f'{domain}: existing raw file is not the pinned release')
        write_json(output.with_suffix('.provenance.json'), {
            'repository': release['repository'], 'commit': release['commit'],
            'url': url, 'sha256': entry['sha256']})
        print(f'Verified {domain}: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', type=Path)
    args = parser.parse_args()
    acquire(Path(__file__).resolve().parents[2], args.local_dir)
