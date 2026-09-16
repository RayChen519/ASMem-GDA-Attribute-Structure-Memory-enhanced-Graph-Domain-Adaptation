"""Fail closed Linux/CUDA preflight using the actual model operators."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import torch
from data.cache.store import write_json

def check(device='cuda:0',require_linux=True):
    if require_linux and platform.system()!='Linux': raise RuntimeError('Server preflight requires Linux')
    from verification.gate.run import require_project_interpreter
    require_project_interpreter()
    if sys.version_info[:2]!=(3,13): raise RuntimeError('Use Python 3.13 for pinned dependencies')
    dependencies={k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','pymetis','pytest')}
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'): raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python')
    torch.use_deterministic_algorithms(True)
    from models.encoders.shared_gcn import SharedGCNEncoder
    from models.selection.operators import PostNormAttention
    from data.partitions.metis import load_partition
    # Actual sparse GCN + full-domain SDPA forward/backward, no dense fallback.
    model=SharedGCNEncoder().to(device)
    x=torch.rand(256,6775,device=device)
    ids=torch.arange(256,device=device)
    adjacency=torch.sparse_coo_tensor(torch.stack((ids,ids)),torch.ones(256,device=device),(256,256)).coalesce()
    z=model(x,adjacency).z; attention=PostNormAttention().to(device)
    attention(z).square().mean().backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()): raise RuntimeError('GCN gradient preflight failed')
    import pymetis
    partition=pymetis.part_graph(2,adjacency=[[1],[0,2],[1,3],[2]],options=pymetis.Options(seed=0))
    if len(set(partition.vertex_part))!=2: raise RuntimeError('METIS preflight failed')
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    return dict(status='PASS',platform=platform.platform(),python=sys.version,dependencies=dependencies,
        cuda=torch.version.cuda,device=device,device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
        free_disk_bytes=shutil.disk_usage(Path.cwd()).free,
        limitation='Operator smoke only; full graph VRAM and long-run CUDA resume require server acceptance')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:0'); p.add_argument('--allow-non-linux',action='store_true'); p.add_argument('--output',required=True); a=p.parse_args()
    result=check(a.device,not a.allow_non_linux); write_json(a.output,result); print(json.dumps(result,indent=2))
