import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from data.cache.store import read_json,write_json
from experiments.configuration import lock_development,configs
from experiments.execution import make_list,verify_list,launch,train
from experiments.registry import FINAL,CORE

def main():
    p=argparse.ArgumentParser(description='Registered experiments: locks, lists, training, resume, evaluation')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('development-run'); a.add_argument('--manifest',required=True); a.add_argument('--integration-report',required=True); a.add_argument('--config',required=True); a.add_argument('--variant',choices=CORE,required=True); a.add_argument('--source',required=True); a.add_argument('--target',required=True); a.add_argument('--rate',type=float,required=True); a.add_argument('--output',default='artifacts/runs/development'); a.add_argument('--cache',default='artifacts/cache'); a.add_argument('--device',default='cuda:0'); a.add_argument('--resume',action='store_true')
    a=sub.add_parser('collect-development'); a.add_argument('--root',required=True); a.add_argument('--output',required=True); a.add_argument('--budget',type=int,default=1)
    a=sub.add_parser('development-template'); a.add_argument('--output',required=True)
    a=sub.add_parser('lock-development'); a.add_argument('--candidate',required=True); a.add_argument('--output',required=True); a.add_argument('--integration-report',required=True)
    a=sub.add_parser('list'); a.add_argument('--tier',choices=['core','ablation','sensitivity'],required=True); a.add_argument('--manifest',required=True); a.add_argument('--development',required=True); a.add_argument('--integration-report',required=True); a.add_argument('--output',required=True)
    for command in ('run','batch'):
        a=sub.add_parser(command); a.add_argument('--list',required=True); a.add_argument('--root',default='artifacts/runs/formal'); a.add_argument('--cache',default='artifacts/cache'); a.add_argument('--device',default='cuda:0'); a.add_argument('--resume',action='store_true')
        if command=='run': a.add_argument('--index',type=int,required=True)
        else:
            a.add_argument('--start',type=int,default=0); a.add_argument('--stop',type=int); a.add_argument('--shard',type=int,default=0); a.add_argument('--shards',type=int,default=1)
    a=sub.add_parser('smoke'); a.add_argument('--output',required=True); a.add_argument('--variant',choices=list(FINAL),default='B6'); a.add_argument('--device',default='cpu'); a.add_argument('--resume',action='store_true')
    a=sub.add_parser('evaluate'); a.add_argument('--manifest',required=True); a.add_argument('--run-dir',required=True); a.add_argument('--cache',default='artifacts/cache'); a.add_argument('--device',default='cpu')
    a=sub.add_parser('analyze'); a.add_argument('--root',required=True); a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='development-template':
        write_json(a.output,dict(development_seed=2026,max_trials_per_model=1,models={v:{'global':{},'trials':1,'rates':{},'rate_evidence':{},'evidence':[]} for v in CORE}))
    elif a.command=='collect-development':
        from experiments.development import collect
        collect(a.root,a.output,a.budget)
    elif a.command=='development-run':
        from experiments.development import run
        run(a.manifest,a.integration_report,a.config,a.variant,a.source,a.target,a.rate,a.output,a.cache,a.device,a.resume)
    elif a.command=='lock-development': lock_development(a.candidate,a.output,a.integration_report)
    elif a.command=='list': make_list(a.tier,a.manifest,a.development,a.integration_report,a.output)
    elif a.command=='run': launch(a.list,a.index,a.root,a.cache,a.device,a.resume)
    elif a.command=='batch':
        listing=verify_list(a.list)
        if a.shards<1 or not 0<=a.shard<a.shards: raise ValueError('Invalid shard')
        for index,row in enumerate(listing['rows']):
            if index<a.start or index>=(a.stop if a.stop is not None else len(listing['rows'])) or index%a.shards!=a.shard: continue
            directory=Path(a.root)/listing['tier']/row['run_id']
            if (directory/'run_lock.json').exists():
                from data.cache.store import file_hash,digest
                lock=read_json(directory/'run_lock.json')
                if lock['run_manifest_sha256']!=file_hash(directory/'run_manifest.json') or lock['final_checkpoint']['sha256']!=file_hash(directory/lock['final_checkpoint']['path']): raise ValueError('Corrupt completed run')
                continue
            command=[sys.executable,'-m','experiments','run','--list',a.list,'--index',str(index),'--root',a.root,'--cache',a.cache,'--device',a.device]
            if a.resume: command.append('--resume')
            subprocess.run(command,check=True)  # fail-fast; never retry based on metrics
    elif a.command=='smoke':
        from verification.smoke.synthetic.encoder_fixture import synthetic_dataset
        output=Path(a.output).resolve()
        manifest,*_=synthetic_dataset(output/'fixture',sizes=(256,257))
        row=dict(run_id='smoke__'+a.variant+'__A-to-B__rate-0.05__seed-0',tier='smoke',variant=a.variant,source='A',target='B',label_rate=.05,seed=0,configs={k:asdict(v) for k,v in configs(name=a.variant,smoke=True).items()})
        train(row,manifest,output/'run',output/'cache',device=a.device,resume=a.resume)
    elif a.command=='evaluate':
        from experiments.evaluate import evaluate
        print(json.dumps(evaluate(a.manifest,a.run_dir,a.cache,a.device),indent=2))
    elif a.command=='analyze':
        from experiments.analysis import analyze
        analyze(a.root,a.output)

if __name__=='__main__': main()
