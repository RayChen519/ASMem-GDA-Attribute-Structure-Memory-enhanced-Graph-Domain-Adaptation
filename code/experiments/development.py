"""Source-validation-only development, seed 2026; uniform maximum budget."""
from dataclasses import asdict
from pathlib import Path
import itertools
from data.cache.store import read_json,file_hash,digest
from experiments.configuration import overlay,configs
from experiments.execution import train,DOMAINS
from experiments.registry import CORE
from experiments.scheduling.admission import require_integration
from verification.gate.run import snapshot

def run(manifest,integration,config_path,name,source,target,rate,output,cache,device,resume=False):
    gate=require_integration(integration)
    if gate['dataset_manifest_hash']!=file_hash(manifest): raise ValueError('Gate dataset mismatch')
    if name not in CORE or source==target or source not in DOMAINS or target not in DOMAINS or rate not in (.01,.03,.05): raise ValueError('Development task mismatch')
    candidate=overlay(read_json(config_path))
    row=dict(run_id=f'development__{name}__{source}-to-{target}__rate-{rate}__seed-2026__config-{digest(candidate)[:12]}',
        tier='development',variant=name,source=source,target=target,label_rate=rate,seed=2026,
        configs={k:asdict(v) for k,v in configs(candidate,name=name).items()})
    directory=Path(output)/row['run_id']
    if resume and (directory/'run_lock.json').exists():
        existing=read_json(directory/'run_manifest.json'); lock=read_json(directory/'run_lock.json')
        if existing['code_snapshot_hash']!=snapshot() or existing['resolved_configs']!=row['configs'] or existing['dataset_manifest_sha256']!=file_hash(manifest) or lock['run_manifest_sha256']!=file_hash(directory/'run_manifest.json') or lock['final_checkpoint']['sha256']!=file_hash(directory/lock['final_checkpoint']['path']): raise ValueError('Completed development identity mismatch')
        from training.checkpoints.memory import read_verified
        read_verified(directory/lock['final_checkpoint']['path'],best=True)
        return existing
    return train(row,manifest,directory,cache,device=device,resume=resume,
        provenance=dict(integration_report=str(Path(integration).resolve()),integration_hash=file_hash(integration),development_overlay=candidate))

def verify_evidence(name,row,budget):
    from experiments.configuration import unseal
    expected={(s,t,r) for s,t in itertools.permutations(DOMAINS,2) for r in (.01,.03,.05)}
    trials={}
    for entry in row['evidence']:
        if set(entry)!={'path','sha256'} or file_hash(entry['path'])!=entry['sha256']: raise ValueError('Development evidence hash mismatch')
        path=Path(entry['path']); run=read_json(path); admission=unseal(path.parent/'admission.json')
        lock=read_json(path.parent/'run_lock.json')
        if lock['run_manifest_sha256']!=file_hash(path) or lock['lock_hash']!=digest({k:v for k,v in lock.items() if k!='lock_hash'}): raise ValueError('Development run not locked')
        if run['tier']!='development' or run['split_seed']!=2026 or run['variant']!=name or run['code_snapshot_hash']!=snapshot(): raise ValueError('Invalid development evidence')
        if any(c['smoke'] for c in run['resolved_configs'].values()): raise ValueError('Smoke is not development evidence')
        from training.checkpoints.memory import read_verified
        final=read_verified(path.parent/lock['final_checkpoint']['path'],best=True)
        if file_hash(path.parent/lock['final_checkpoint']['path'])!=lock['final_checkpoint']['sha256']: raise ValueError('Development checkpoint changed')
        config=admission['provenance']['development_overlay']; key=digest(config)
        if run['resolved_configs']!={k:asdict(v) for k,v in configs(config,name=name).items()}: raise ValueError('Development resolved config mismatch')
        trial=trials.setdefault(key,dict(config=config,conditions={}))
        condition=(run['source'],run['target'],run['label_rate'])
        if condition in trial['conditions']: raise ValueError('Duplicate development condition')
        trial['conditions'][condition]=(final['early_stop_state']['best_macro_f1'],final['early_stop_state']['best_loss'])
    if len(trials)!=row['trials'] or not 1<=len(trials)<=budget or any(set(t['conditions'])!=expected for t in trials.values()): raise ValueError('Complete 18-condition development evidence required for every trial')
    def score(item):
        key,t=item; metrics=list(t['conditions'].values())
        return (-sum(v[0] for v in metrics)/18,sum(v[1] for v in metrics)/18,key)
    winner=min(trials.items(),key=score)[1]
    if row['global']!=winner['config']: raise ValueError('Global config must win mean Source validation, loss tie-break')
    for rate,config in row['rates'].items():
        candidates=[t for t in trials.values() if t['config']==config]
        if len(candidates)!=1: raise ValueError('Rate config lacks complete development evidence')
        conditions=sorted(c for c in expected if c[2]==float(rate))
        delta=[candidates[0]['conditions'][c][0]-winner['conditions'][c][0] for c in conditions]
        if row['rate_evidence'].get(rate,{}).get('source_validation_deltas')!=delta: raise ValueError('Rate evidence must match measured Source validation deltas')


def collect(root,output,budget=1):
    from experiments.configuration import unseal
    from data.cache.store import write_json
    grouped={v:{} for v in CORE}
    for path in sorted(Path(root).glob('*/run_manifest.json')):
        run=read_json(path)
        if run['tier']!='development': continue
        admission=unseal(path.parent/'admission.json')
        config=admission['provenance']['development_overlay']
        trial=grouped[run['variant']].setdefault(digest(config),dict(config=config,rows=[]))
        trial['rows'].append((path,run))
    models={}
    for name,trials in grouped.items():
        if not trials: raise ValueError('Missing development model '+name)
        def score(item):
            key,trial=item
            metrics=[r['checkpoint_lineage'][-1]['source_validation'] for _,r in trial['rows']]
            return (-sum(m['best_macro_f1'] for m in metrics)/len(metrics),sum(m['best_loss'] for m in metrics)/len(metrics),key)
        winner=min(trials.items(),key=score)[1]
        row={'global':winner['config'],'trials':len(trials),'rates':{},'rate_evidence':{},
             'evidence':[dict(path=str(p.resolve()),sha256=file_hash(p)) for t in trials.values() for p,_ in t['rows']]}
        verify_evidence(name,row,budget); models[name]=row
    write_json(output,dict(development_seed=2026,max_trials_per_model=budget,models=models))
