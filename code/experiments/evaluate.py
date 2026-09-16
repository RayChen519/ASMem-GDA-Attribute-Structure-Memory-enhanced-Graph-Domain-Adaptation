"""Read-only final inference; only this module opens Target evaluation labels."""
from pathlib import Path
import time
import torch
from data.cache.store import read_json,write_json,file_hash,digest
from data.views.training import load_manifest,verified_tensor
from contracts.data import TargetTrainView
from data.partitions.metis import load_partition
from evaluation.final_target.data import load_target_evaluation
from evaluation.metrics.classification import classification_metrics
from experiments.registry import variant
from models.encoders.shared_gcn import SharedGCNEncoder
from models.classifiers.shared import SharedClassifier
from models.fusion.attribute_structure import AttributeStructure
from training.checkpoints.memory import read_verified
from training.stages.encoder_warmup.trainer import graph_inputs,evaluation_mode
from verification.gate.run import snapshot

def evaluate(manifest,run_dir,cache,device='cpu'):
    run_dir=Path(run_dir)
    if (run_dir/'.running').exists(): raise PermissionError('Training process still active')
    run=read_json(run_dir/'run_manifest.json'); lock=read_json(run_dir/'run_lock.json')
    if run['tier']=='development': raise PermissionError('Development has no Target evaluation')
    if run.get('code_snapshot_hash')!=snapshot(): raise ValueError('Code changed since training; invalidate run')
    if lock['run_manifest_sha256']!=file_hash(run_dir/'run_manifest.json') or lock['lock_hash']!=digest({k:v for k,v in lock.items() if k!='lock_hash'}): raise ValueError('Invalid final lock')
    cp=run_dir/lock['final_checkpoint']['path']
    if not cp.resolve().is_relative_to(run_dir.resolve()) or file_hash(cp)!=lock['final_checkpoint']['sha256']: raise ValueError('Final checkpoint mismatch')
    payload=read_verified(cp,best=True)
    root,data=load_manifest(manifest)
    graph=verified_tensor(root,data['domains'][run['target']]['graph'])
    target=TargetTrainView(**{k:graph[k] for k in TargetTrainView.__dataclass_fields__})
    encoder=SharedGCNEncoder().to(device); classifier=SharedClassifier(len(data['class_map'])).to(device)
    spec=variant(run['variant']); modules=[encoder,classifier]
    if spec['stages']==1:
        encoder.load_state_dict(payload['shared_gcn_state']); classifier.load_state_dict(payload['classifier_state'])
    else:
        model=AttributeStructure(spec['options']).to(device); modules.append(model)
        for name,module in [('shared_gcn',encoder),('classifier',classifier),('attribute_structure',model)]: module.load_state_dict(payload['model_state'][name])
        partition=load_partition(target,run['target'],cache,P=run['resolved_configs']['da']['P'])
    started=time.perf_counter()
    with evaluation_mode(*modules):
        z=encoder(*graph_inputs(target,device)).z
        logits=classifier(z if spec['stages']==1 else model(z,partition,source=False).h_as).cpu()
    elapsed=time.perf_counter()-started
    # Inference and every lock check precede label access.
    evaluation=load_target_evaluation(manifest,run_dir/'run_manifest.json',run_dir/'run_lock.json')
    if not torch.equal(target.node_id,evaluation.node_id): raise ValueError('Target node order mismatch')
    result=dict(status='COMPLETED',run_id=run['run_id'],variant=run['variant'],tier=run['tier'],
        source=run['source'],target=run['target'],label_rate=run['label_rate'],seed=run['split_seed'],
        checkpoint_hash=evaluation.checkpoint_hash,run_manifest_hash=evaluation.run_manifest_hash,
        evaluation_code_hash=snapshot(),inference_seconds=elapsed,
        metrics=classification_metrics(logits,evaluation.y,len(data['class_map'])))
    result['result_hash']=digest(result)
    write_json(run_dir/'evaluation.json',result)
    write_json(run_dir/'status.json',dict(status='COMPLETED',evaluation_hash=file_hash(run_dir/'evaluation.json')))
    with (run_dir/'target_access.jsonl').open('a',encoding='utf8') as f:
        import json
        f.write(json.dumps(dict(checkpoint_hash=evaluation.checkpoint_hash,operation='final_labels',evaluation_hash=result['result_hash']))+'\n')
    return result

def verified_result(directory):
    directory=Path(directory); status=read_json(directory/'status.json'); result=read_json(directory/'evaluation.json')
    if status.get('status')!='COMPLETED' or status['evaluation_hash']!=file_hash(directory/'evaluation.json'): raise ValueError('Run not COMPLETED')
    if result['result_hash']!=digest({k:v for k,v in result.items() if k!='result_hash'}): raise ValueError('Evaluation integrity failure')
    run=read_json(directory/'run_manifest.json'); lock=read_json(directory/'run_lock.json')
    if result['run_manifest_hash']!=file_hash(directory/'run_manifest.json') or result['checkpoint_hash']!=file_hash(directory/lock['final_checkpoint']['path']): raise ValueError('Result lineage changed')
    if lock['lock_hash']!=digest({k:v for k,v in lock.items() if k!='lock_hash'}) or lock['run_manifest_sha256']!=result['run_manifest_hash'] or lock['final_checkpoint']['sha256']!=result['checkpoint_hash']: raise ValueError('Result lock mismatch')
    if result['evaluation_code_hash']!=snapshot() or run['code_snapshot_hash']!=snapshot(): raise ValueError('Stale result')
    read_verified(directory/lock['final_checkpoint']['path'],best=True)
    return result,run
