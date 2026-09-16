"""Five-stage orchestration. Training and Target evaluation are separate processes."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import torch
from data.cache.store import digest, file_hash, read_json, write_json
from experiments.registry import CORE, ABLATIONS, variant, registry_hash
from experiments.configuration import configs, resolved, seal, unseal, read_development_lock
from experiments.scheduling.admission import require_integration
from verification.gate.run import snapshot

DOMAINS=("ACMv9","Citationv1","DBLPv7")
SENSITIVITY={"K":(64,256),"temperature":(.05,.2),"gamma":(.85,.95),"q":(.1,.3),"P":(64,256)}

def inventory(tier, defaults=None):
    names=CORE if tier=="core" else ABLATIONS if tier=="ablation" else ("B6",) if tier=="sensitivity" else ()
    if not names: raise ValueError("Unknown tier")
    defaults=defaults or {}
    factors_spec=dict(SENSITIVITY)
    for key,default in {'K':128,'temperature':.1,'gamma':.9,'q':.2,'P':128}.items():
        grid=sorted((*SENSITIVITY[key],default))
        selected=defaults.get('memory',{}).get(key,default) if key!='P' else 128
        if selected not in grid: raise ValueError('Sensitivity reference must belong to preregistered grid')
        factors_spec[key]=tuple(v for v in grid if v!=selected)
    rows=[]
    for name,(source,target),rate,seed in itertools.product(names,itertools.permutations(DOMAINS,2),(.03,) if tier=="sensitivity" else (.01,.03,.05),range(3 if tier=="sensitivity" else 5)):
        factors=[(k,v) for k,values in factors_spec.items() for v in values] if tier=="sensitivity" else [None]
        for factor in factors:
            identity=f"{tier}__{name}__{source}-to-{target}__rate-{rate}__seed-{seed}"
            if factor: identity+=f"__{factor[0]}-{factor[1]}"
            rows.append(dict(run_id=identity,tier=tier,variant=name,source=source,target=target,label_rate=rate,seed=seed,sensitivity=factor))
    return rows

def asset_identity(row,manifest,cache,memo=None):
    from data.views.training import load_source_training,load_manifest,verified_tensor
    from contracts.data import TargetTrainView
    from data.partitions.metis import load_partition
    from data.anchors.source import sample_anchors
    from data.prepare import tensor_hash
    memo={} if memo is None else memo
    source_name,target_name,rate,seed=[row[k] for k in ('source','target','label_rate','seed')]
    key=(source_name,rate,seed)
    if memo.get('source_key')!=key:
        memo['source']=load_source_training(manifest,source_name,rate,seed); memo['source_key']=key
    source=memo['source']; root,data=load_manifest(manifest); spec=variant(row['variant'])
    identity=dict(split_hash=source.split_hash,attribute_union_hash=source.graph.attribute_union_hash,
        source_graph_hash=data['domains'][source_name]['graph']['sha256'],
        anchor_candidate_pool_hash=tensor_hash({'ids':source.graph.node_id[source.graph.source_unlabeled_mask]}))
    if spec['stages']>1:
        if target_name not in memo:
            g=verified_tensor(root,data['domains'][target_name]['graph'])
            memo[target_name]=TargetTrainView(**{k:g[k] for k in TargetTrainView.__dataclass_fields__})
        target=memo[target_name]; partitions={}
        for name,g in ((source_name,source.graph),(target_name,target)):
            pk=(name,row['configs']['da']['P'])
            if pk not in memo: memo[pk]=load_partition(g,name,cache,P=pk[1]).fingerprint
            partitions[name]=memo[pk]
        identity.update(partitions=partitions,target_graph_hash=data['domains'][target_name]['graph']['sha256'],target_order_hash=tensor_hash({'ids':target.node_id}))
    if spec['stages']>2:
        c=row['configs']['memory_warmup']
        anchor=sample_anchors(source,dict(source=source_name,target=target_name,label_rate=rate,seed=seed),cache,
            K=c['K'],alpha=c['alpha'],beta=c['beta'],strategy=spec['options']['anchors'])
        identity.update(anchor_hash=anchor['hash'],anchor_inputs_hash=anchor['inputs_hash'])
    return identity


def make_list(tier, manifest, development, integration, output):
    gate=require_integration(integration)
    if gate['dataset_manifest_hash']!=file_hash(manifest): raise ValueError("Gate dataset mismatch")
    development_lock=read_development_lock(development)
    rows=inventory(tier,resolved(development,"B6",.03,development_lock) if tier=="sensitivity" else None)
    memo={}
    from verification.gate.run import ROOT
    for row in rows:
        overrides=resolved(development,row['variant'],row['label_rate'],development_lock)
        cfg=configs(overrides,name=row['variant'],sensitivity=row['sensitivity'])
        row['configs']={k:asdict(v) for k,v in cfg.items()}
        row['config_hash']=digest(row['configs'])
        row['assets']=asset_identity(row,manifest,ROOT/'artifacts/cache',memo)
    seal(output,dict(schema='experiment-list-v1',tier=tier,rows=rows,
        manifest=str(Path(manifest).resolve()),dataset_manifest_sha256=file_hash(manifest),
        development=str(Path(development).resolve()),development_hash=file_hash(development),
        integration=str(Path(integration).resolve()),integration_hash=file_hash(integration),
        code_snapshot_hash=snapshot(),registry_hash=registry_hash()))

def verify_list(path):
    listing=unseal(path)
    if listing['code_snapshot_hash']!=snapshot() or listing['registry_hash']!=registry_hash(): raise ValueError("Stale run list")
    for key,hashkey in [('manifest','dataset_manifest_sha256'),('development','development_hash'),('integration','integration_hash')]:
        if file_hash(listing[key])!=listing[hashkey]: raise ValueError("Changed locked input: "+key)
    gate=require_integration(listing['integration'])
    if gate['dataset_manifest_hash']!=listing['dataset_manifest_sha256']: raise ValueError('Gate dataset mismatch')
    development_lock=read_development_lock(listing['development'])
    expected=inventory(listing['tier'],resolved(listing['development'],'B6',.03,development_lock) if listing['tier']=='sensitivity' else None)
    if len(expected)!=len(listing['rows']): raise ValueError('Incomplete list')
    for row, spec in zip(listing['rows'],expected):
        if any(row[k]!=v and not (k=='sensitivity' and row[k]==list(v or [])) for k,v in spec.items()): raise ValueError('Noncanonical run list')
        actual={k:asdict(v) for k,v in configs(resolved(listing['development'],row['variant'],row['label_rate'],development_lock),name=row['variant'],sensitivity=row['sensitivity']).items()}
        if row['configs']!=actual or row['config_hash']!=digest(actual): raise ValueError('Resolved configuration mismatch')
    return listing

def runtime(device):
    import importlib.metadata
    return dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
        device=str(device),device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else platform.processor(),
        host=platform.node(),pid=os.getpid(),dependencies={k:importlib.metadata.version(k) for k in ('numpy','scipy','pymetis','pytest')})

def train(row,manifest,output,cache,*,device='cpu',resume=False,provenance=None):
    from contracts.training.encoder import WarmupConfig
    from contracts.training.domain_adaptation import DAConfig
    from contracts.training.memory import MemoryConfig
    from data.views.training import load_training_views,load_source_training
    from data.partitions.metis import load_partition
    from evaluation.source_validation.data import load_source_validation
    from training.checkpoints.encoder import build_metadata,load_checkpoint
    from training.checkpoints.domain_adaptation import load_checkpoint as load_da
    from training.checkpoints.memory import read_verified,validate
    from training.stages.encoder_warmup.trainer import EncoderWarmup
    from training.stages.domain_adaptation.trainer import DomainAdaptation
    from training.stages.memory_warmup.trainer import MemoryTraining
    from models.encoders.shared_gcn import SharedGCNEncoder
    from models.classifiers.shared import SharedClassifier
    from utils.randomness.state import seed_everything
    smoke=all(c.get('smoke') is True for c in row['configs'].values())
    if smoke:
        if row['tier']!='smoke': raise PermissionError('Smoke cannot enter formal/development tiers')
    elif row['tier'] in ('core','ablation','sensitivity'):
        if not provenance or 'list_path' not in provenance: raise PermissionError('Formal training requires locked list admission')
        listing=verify_list(provenance['list_path'])
        if listing['rows'][provenance['index']]!=row or listing['dataset_manifest_sha256']!=file_hash(manifest): raise PermissionError('Formal task differs from locked list')
    elif row['tier']=='development':
        if not provenance or 'integration_report' not in provenance: raise PermissionError('Development requires Integration PASS')
        gate=require_integration(provenance['integration_report'])
        if gate['dataset_manifest_hash']!=file_hash(manifest) or row['seed']!=2026 or row['variant'] not in CORE: raise PermissionError('Invalid development admission')
    else: raise PermissionError('Unregistered training tier')
    if not smoke:
        gate=require_integration(listing['integration'] if row['tier']!='development' else provenance['integration_report'])
        if gate['environment']['device'].split(':')[0]!=str(device).split(':')[0]: raise PermissionError('Integration device type differs from training')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if (output/'run_lock.json').exists(): raise ValueError('Training already locked; use independent evaluation')
    env=runtime(device)
    if not smoke and (env['torch']!=gate['environment']['torch'] or env['dependencies']!={k:v for k,v in gate['environment']['dependencies'].items() if k!='torch'}): raise PermissionError('Integration dependency environment changed')
    identity=dict(execution_environment={k:env[k] for k in ('python','torch','cuda','device','device_name','dependencies')},row=row,dataset_manifest_sha256=file_hash(manifest),code_snapshot_hash=snapshot(),provenance=provenance)
    admission=output/'admission.json'
    if admission.exists():
        if not resume or unseal(admission)!=identity: raise ValueError('Resume identity mismatch or explicit resume missing')
    else:
        if any(p.name not in ('.running','attempts.jsonl') for p in output.iterdir()): raise ValueError('New run directory must be empty')
        seal(admission,identity)
    if 'assets' in row and asset_identity(row,manifest,cache)!=row['assets']: raise ValueError('Locked data/METIS/anchor assets changed')
    source_name,target_name,rate,seed=[row[k] for k in ('source','target','label_rate','seed')]
    spec=variant(row['variant'])
    torch.use_deterministic_algorithms(True); seed_everything(seed)
    torch.set_num_threads(2)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
    if spec['stages']==1:
        source=load_source_training(manifest,source_name,rate,seed); target=None
    else: source,target=load_training_views(manifest,source_name,target_name,rate,seed)
    validation=load_source_validation(manifest,source_name,rate,seed)
    cfg={k:(WarmupConfig if k=='encoder' else DAConfig if k=='da' else MemoryConfig)(**v) for k,v in row['configs'].items()}
    meta=build_metadata(manifest,source_name,target_name,rate,seed,source,target,cfg['encoder'])
    meta['experiment_identity_hash']=digest(identity)
    warm=EncoderWarmup(SharedGCNEncoder(),SharedClassifier(meta['num_classes']),source,validation,meta,output/'encoder',cfg['encoder'],device)
    lineage=[]
    def stage(trainer,folder,best):
        directory=output/folder; cp=directory/(best+'.pt')
        marker=directory/'completed.json'
        if marker.exists():
            record=unseal(marker)
            if record['checkpoint_hash']!=file_hash(cp): raise ValueError('Completed stage checkpoint changed')
        else:
            latest=directory/(best.replace('_best','_latest')+'.pt')
            if latest.exists():
                if not resume: raise ValueError('Explicit resume required')
                trainer.resume(latest)
            elif cp.exists(): raise ValueError('Incomplete checkpoint transaction; audit and rerun required')
            timing_path=directory/'timing.json'
            timing=read_json(timing_path) if timing_path.exists() else dict(training_seconds=0.,pl_refresh_seconds=0.)
            if hasattr(trainer,'refresh'):
                refresh=trainer.refresh
                def timed_refresh(epoch):
                    before=time.perf_counter()
                    result=refresh(epoch)
                    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
                    timing['pl_refresh_seconds']+=time.perf_counter()-before
                    return result
                trainer.refresh=timed_refresh
            started=time.perf_counter()
            try: trainer.fit()
            finally:
                if str(device).startswith('cuda'): torch.cuda.synchronize(device)
                timing['training_seconds']+=time.perf_counter()-started
                write_json(timing_path,timing)
            record=dict(stage=folder,checkpoint_hash=file_hash(cp),completed_epochs=trainer.epoch,**timing)
            seal(marker,record)
        payload=read_verified(cp,best=True)
        modules=getattr(trainer,'modules',{'shared_gcn':warm.encoder,'classifier':warm.classifier})
        if folder in ('memory','source','dual'):
            validate(payload,modules,trainer.metadata); trainer._load_models(payload)
        elif folder=='encoder': load_checkpoint(cp,warm.encoder,warm.classifier,meta)
        else: load_da(cp,modules,trainer.metadata)
        record.update(path=str(cp.relative_to(output)),name=best,best_epoch=payload['epoch'],
                      source_validation=payload['early_stop_state'] if payload['epoch'] else None,metadata=payload['metadata'],
                      checkpoint_bytes=cp.stat().st_size,total_parameters=sum(p.numel() for m in modules.values() for p in m.parameters()),
                      trainable_parameters=sum(p.numel() for m in modules.values() for p in m.parameters() if p.requires_grad))
        from utils.randomness.state import restore_rng
        latest=cp.with_name(best.replace('_best','_latest')+'.pt')
        if best!='untrained_memory' and not latest.exists(): raise ValueError('Completed stage is missing latest RNG state')
        restore_rng(read_verified(latest if latest.exists() else cp)['rng_state'])
        lineage.append(record)
        return cp
    ep=stage(warm,'encoder','encoder_best'); final=ep
    if spec['stages']>1:
        parts=[load_partition(g,n,cache,P=cfg['da'].P) for g,n in ((source.graph,source_name),(target,target_name))]
        da=DomainAdaptation(warm.encoder,warm.classifier,source,target,validation,ep,meta,*parts,output/'da',cfg['da'],device)
        dp=stage(da,'da','da_best'); final=dp
    memory_best=output/'memory'/('memory_best.pt' if spec['options']['memory_warmup'] else 'untrained_memory.pt')
    for index,(name,folder,best) in enumerate((('memory_warmup','memory',memory_best.stem),('source_self_training','source','source_pl_best'),('dual_domain_finetuning','dual','full_best')),3):
        if index>spec['stages']: break
        trainer=MemoryTraining(da,source,dp,output/folder,cfg[name],cache,memory_best=memory_best,source_pl_best=output/'source/source_pl_best.pt')
        final=stage(trainer,folder,best)
    if final.stem!=spec['final']: raise ValueError('Wrong final stage')
    if snapshot()!=identity['code_snapshot_hash'] or file_hash(manifest)!=identity['dataset_manifest_sha256']: raise ValueError('Code/data changed during training; invalidate and rerun')
    run=dict(run_id=row['run_id'],tier=row['tier'],variant=row['variant'],parent=spec['parent'],changed_component=spec['changed_component'],
        source=source_name,target=target_name,label_rate=rate,split_seed=seed,
        assets=row.get('assets'),admission_hash=file_hash(admission),resolved_configs=row['configs'],config_hash=digest(row['configs']),
        dataset_version=meta['dataset_version'],dataset_manifest_sha256=file_hash(manifest),split_hash=meta['split_hash'],
        attribute_union_hash=meta['attribute_union_hash'],code_snapshot_hash=snapshot(),registry_hash=registry_hash(),
        anchor_hash=lineage[-1]['metadata'].get('anchor_hash'),metis_hash=lineage[-1]['metadata'].get('metis_hash'),
        checkpoint_lineage=lineage,environment=runtime(device),finished_utc=datetime.now(timezone.utc).isoformat(),
        training_seconds=sum(r['training_seconds'] for r in lineage),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None)
    write_json(output/'run_manifest.json',run)
    lock=dict(status='locked',run_manifest_sha256=file_hash(output/'run_manifest.json'),
              final_checkpoint=dict(name=final.stem,path=str(final.relative_to(output)),sha256=file_hash(final)))
    write_json(output/'run_lock.json',{**lock,'lock_hash':digest(lock)})
    write_json(output/'status.json',dict(status='TRAINED',run_id=row['run_id']))
    return run

def launch(list_path,index,root,cache,device,resume):
    listing=verify_list(list_path)
    row=listing['rows'][index]
    output=Path(root)/listing['tier']/row['run_id']; output.mkdir(parents=True,exist_ok=True)
    # Exclusive file creation prevents duplicate scheduler workers on shared storage.
    mutex=output/'.running'
    with mutex.open('x',encoding='utf8') as f: f.write(str(os.getpid()))
    try:
        return train(row,listing['manifest'],output,cache,device=device,resume=resume,
                     provenance=dict(list_path=str(Path(list_path).resolve()),list_hash=file_hash(list_path),index=index))
    except BaseException as exc:
        with (output/'attempts.jsonl').open('a',encoding='utf8') as f:
            import json
            f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),error=repr(exc),traceback=traceback.format_exc()))+'\n')
        raise
    finally: mutex.unlink()
