"""Require all feature/split caches to hit and all manifest bytes to reproduce."""
from pathlib import Path

from data.cache.store import Cache, file_hash, read_json, write_json
from data.prepare import prepare


def reproduce(root):
    root=Path(root).resolve()
    latest=root/'artifacts/datasets/manifests/dataset_manifest.json'
    previous=file_hash(latest)
    manifest=read_json(latest)
    original=Cache.get_or_create
    hits={'features':0,'split':0}
    def require_hit(self,kind,inputs,factory):
        def forbid_factory():
            raise AssertionError(f'Expected existing cache for {kind}')
        result=original(self,kind,inputs,forbid_factory)
        assert result[2]
        hits[kind]+=1
        return result
    Cache.get_or_create=require_hit
    try:
        rebuilt=prepare(root)
    finally:
        Cache.get_or_create=original
    assert file_hash(rebuilt) == previous == file_hash(latest), 'Rebuilt manifest differs'
    assert hits == {'features':3,'split':54}
    report={'status':'PASS','dataset_version':manifest['dataset_version'],
            'manifest_sha256':previous,'byte_identical_manifest':True,'cache_hits':hits,
            'referenced_artifact_hashes_identical':True}
    write_json(root/'artifacts/reports/integration/dataset_reproducibility.json',report)
    print(f'PASS: deterministic rebuild, cache hits {hits}')
    return report


if __name__ == '__main__':
    reproduce(Path(__file__).resolve().parents[2])
