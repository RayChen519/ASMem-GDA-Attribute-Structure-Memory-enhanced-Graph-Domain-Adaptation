"""Ordered, fail-closed P6 acceptance: python -m verification.gate.run."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import torch
from data.cache.store import digest, file_hash, read_json, write_json

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parent
REQUIRED=('dimension_contract','gradient_whitelist','grl_single_reversal','target_label_sentinel',
          'memory_teacher_frozen','pseudo_label_detach','refresh_and_resume','checkpoint_lineage','smoke_final_evaluation')


def snapshot():
    files=[p for folder in ('contracts','data','models','training','evaluation','experiments','configs','verification','utils')
           for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    files+=list((REPO/'plan').glob('*.md'))+[ROOT/'Framework_intro.md']
    files += list(ROOT.glob('requirements*.txt'))
    files += [p for p in (REPO/'scripts').glob('*') if p.is_file()]
    if (REPO/'README.md').exists(): files.append(REPO/'README.md')
    return digest({str(p.relative_to(REPO)).replace('\\','/'):file_hash(p) for p in sorted(files)})


def require_pass(path):
    """Formal schedulers must call this before accepting any P6 evidence."""
    r=read_json(path)
    if (r.get('status')!='PASS' or r.get('code_snapshot_hash')!=snapshot()
            or any(r.get(k,{}).get('status')!='PASS' for k in REQUIRED)
            or [c['name'] for c in r.get('checks',[])]!=list(GROUPS)+['synthetic_smoke','full_chain_smoke']
            or any(c['status']!='PASS' for c in r['checks'])):
        raise PermissionError('Integration gate missing, failed, incomplete or stale')
    for path_,hash_ in r['evidence_hashes'].items():
        if file_hash(path_)!=hash_: raise PermissionError('Integration evidence changed')
    return r


def require_project_interpreter(prefix=None, base_prefix=None, repo=REPO):
    """Accept the repository venv on both Windows and POSIX hosts."""
    prefix = sys.prefix if prefix is None else prefix
    base_prefix = sys.base_prefix if base_prefix is None else base_prefix
    if Path(prefix).resolve() != (Path(repo) / '.venv').resolve() or prefix == base_prefix:
        raise RuntimeError('Use the project .venv interpreter')


E='verification/checkpoint_resume/test_encoder.py'
D='verification/checkpoint_resume/test_domain_adaptation_resume.py'
M='verification/checkpoint_resume/test_memory_resume.py'
DIM='verification/dimensions/test_domain_adaptation.py'
GROUPS={
 'static_config':['verification/static_config','verification/gate/test_admission.py',DIM+'::test_config_files_and_full_optimizer_schedule'],
 'dimensions':['verification/dimensions/test_dimensions_integration.py',
     DIM+'::test_selection_exact_formula_postnorm_threshold_and_gradients',
     DIM+'::test_full_domain_attention_sees_remote_node',
     DIM+'::test_metis_cache_keys_graph_integrity_and_order',
     DIM+'::test_bga_equations_restore_order_sharing_and_target_bypass',
     E+'::test_shared_parameters_shapes_initialization_and_exact_adjacency'],
 'stage_gradients':[E+'::test_source_only_gradients_and_validation_isolation',D+'::test_inherit_joint_gradients_optimizer_and_no_memory',
                    M+'::test_exact_resume_and_frozen_parameters',M+'::test_epoch20_21_gradients_refresh_and_teacher',
                    M+'::test_nonempty_pseudo_labels_train_classifier_only'],
 'grl':[DIM+'::test_single_grl_and_weight_applied_once'],
 'target_sentinel':[E+'::test_evaluation_file_replacements_leave_entire_training_state_identical',
                    D+'::test_target_sentinel_and_evaluation_file_changes_do_not_affect_training',
                    M+'::test_target_label_sentinel_all_three_stages','verification/dataset/test_dataset.py'],
 'pseudo_label_refresh':['verification/memory_refresh','verification/checkpoint_resume'],
 'registered_experiments':['verification/experiments']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'artifacts/datasets/manifests/dataset_manifest.json')
    p.add_argument('--config',type=Path,default=ROOT/'configs/overrides/smoke/integration.json')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    require_project_interpreter()
    out,runs=a.output.resolve(),a.runs.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Report output must be empty')
    if runs.exists() and any(runs.iterdir()): raise ValueError('Run output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    runs.mkdir(parents=True,exist_ok=True)
    report={'schema_version':'integration-v1','status':'RUNNING','formal_experiments_allowed':False,
            'command':subprocess.list2cmdline([sys.executable,'-m','verification.gate.run',*sys.argv[1:]]),
            'started_utc':datetime.now(timezone.utc).isoformat(),'code_snapshot_hash':snapshot(),
            'dataset_manifest':str(a.manifest.resolve()),'dataset_manifest_hash':file_hash(a.manifest),
            'config':read_json(a.config),'config_hash':file_hash(a.config),'checks':[],
            'environment':{'python':sys.executable,'torch':str(torch.__version__),'cuda':torch.cuda.is_available(),
                'device':a.device,'host':platform.node(),'pid':os.getpid(),
                'dependencies':{k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','pymetis','pytest')}},
            'evidence_hashes':{},**{k:{'status':'NOT_RUN'} for k in REQUIRED}}
    path=out/'integration_report.json'
    def persist(): write_json(path,report)
    persist()
    try:
        for index,(name,selectors) in enumerate(GROUPS.items()):
            log,xml=out/(name+'.log'),out/(name+'.xml')
            # Short unique temp roots avoid Windows MAX_PATH in Dataset regression.
            temp=REPO/('pytest-tmp-gate-'+out.name[-12:]+'-'+str(index))
            cmd=[sys.executable,'-m','pytest',*selectors,'-q','-p','no:cacheprovider','--basetemp='+str(temp),'--junitxml='+str(xml)]
            row={'name':name,'status':'RUNNING','command':subprocess.list2cmdline(cmd),'log':str(log),'junit':str(xml)}
            report['checks'].append(row); persist()
            print(name+' START',flush=True)
            start=time.perf_counter()
            with log.open('w',encoding='utf-8') as f:
                result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            row.update(exit_code=result.returncode,elapsed_seconds=time.perf_counter()-start)
            cases=ET.parse(xml).findall('.//testcase') if xml.exists() else []
            row['test_cases']=[{'id':c.get('classname')+'::'+c.get('name'),
                 'status':'FAIL' if any(c.find(k) is not None for k in ('failure','error','skipped')) else 'PASS'} for c in cases]
            row['status']='PASS' if result.returncode==0 and cases and all(c['status']=='PASS' for c in row['test_cases']) else 'FAIL'
            persist(); print(name+' '+row['status'],flush=True)
            if row['status']!='PASS': raise RuntimeError('Required check failed: '+name)
        from verification.smoke.synthetic.encoder_fixture import synthetic_dataset
        from verification.smoke.full_chain.run import run
        for name in ('synthetic_smoke','full_chain_smoke'):
            row={'name':name,'status':'RUNNING'}; report['checks'].append(row); persist()
            print(name+' START',flush=True)
            manifest=a.manifest
            kw={}
            if name=='synthetic_smoke':
                manifest,*_=synthetic_dataset(runs/'synthetic_data',sizes=(256,257))
                kw={'source_name':'A','target_name':'B'}
            r=run(manifest,runs/name,a.config,device=a.device,**kw)
            row.update(status=r['status'],report=str(runs/name/'chain_report.json'))
            persist()
        if report['code_snapshot_hash'] != snapshot():
            raise RuntimeError('Code/config changed during acceptance; rerun required')
        mapping={'dimension_contract':['static_config','dimensions'], 'gradient_whitelist':['stage_gradients'],
                 'grl_single_reversal':['grl'], 'target_label_sentinel':['target_sentinel'],
                 'memory_teacher_frozen':['stage_gradients','pseudo_label_refresh'],
                 'pseudo_label_detach':['pseudo_label_refresh'], 'refresh_and_resume':['pseudo_label_refresh','synthetic_smoke','full_chain_smoke'],
                 'checkpoint_lineage':['synthetic_smoke','full_chain_smoke'],
                 'smoke_final_evaluation':['synthetic_smoke','full_chain_smoke']}
        for key in REQUIRED: report[key]={'status':'PASS','evidence':mapping[key]}
        report['status']='PASS'
        report['formal_experiments_allowed']=True
        # Binding data, code/config snapshot, checkpoints, manifests, locks and test evidence.
        evidence=[a.manifest,a.config]+list(out.glob('*.log'))+list(out.glob('*.xml'))
        evidence += [p for p in runs.rglob('*') if p.is_file() and p.suffix in ('.json','.pt')]
        report['evidence_hashes']={str(p.resolve()):file_hash(p) for p in evidence}
    except BaseException as exc:
        report['status']='FAIL'; report['formal_experiments_allowed']=False
        report['error']=repr(exc); report['traceback']=traceback.format_exc()
        if report['checks'] and report['checks'][-1]['status']=='RUNNING': report['checks'][-1]['status']='FAIL'
        print(report['traceback'],flush=True)
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    persist()
    (out/'README.md').write_text('# Integration '+report['status']+'\n\nCommand: `'+report['command']+'`\n\n'
        +'Device: '+a.device+'; snapshot: '+report['code_snapshot_hash']+'\n\n'
        +'\n'.join('- '+c['name']+': '+c['status'] for c in report['checks'])+'\n\n'
        +report.get('error','All required checks executed; performance is not a gate criterion.')+'\n',encoding='utf-8')
    if report['status']=='PASS': require_pass(path)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': sys.exit(main())
