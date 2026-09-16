"""Pre-specified complete-case statistics; missing seeds never become formal means."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon,rankdata
from data.cache.store import write_json,digest
from experiments.registry import CORE,ABLATIONS
from experiments.evaluate import verified_result

PROTOCOL=dict(version='statistics-v1',primary='macro_f1',secondary='accuracy',seeds=list(range(5)),
    conditions=18,dispersion='sample_sd_ddof_1',test='paired_two_sided_wilcoxon',
    correction='Holm_separate_core_and_ablation_families_per_endpoint',alpha=.05,
    bootstrap='resample_conditions_then_paired_seeds_within_condition',replicates=10000,seed=2026)

def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); adjusted=np.empty_like(p)
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted.tolist()

def bootstrap(delta,replicates=10000,seed=2026,statistic='mean'):
    delta=np.asarray(delta,float); rng=np.random.default_rng(seed)
    conditions,seeds=delta.shape; values=[]
    for _ in range(replicates):
        c=rng.integers(conditions,size=conditions); s=rng.integers(seeds,size=(conditions,seeds))
        sampled=delta[c[:,None],s]
        values.append(float(np.mean(sampled<0)) if statistic=='negative_fraction' else float(np.mean(sampled)))
    return np.quantile(values,[.025,.975]).tolist()

def paired(a,b):
    delta=np.asarray(a)-np.asarray(b); means=delta.mean(1)
    nonzero=means[means!=0]; ranks=rankdata(abs(nonzero))
    effect=float(np.sum(ranks*np.sign(nonzero))/ranks.sum()) if len(nonzero) else 0.
    return dict(raw_p=float(wilcoxon(means,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if len(nonzero) else 1.,
        rank_biserial=effect,paired_median_difference=float(np.median(means)),mean_difference=float(means.mean()),
        bootstrap_95_ci=bootstrap(delta))

def analyze(root,output):
    groups=defaultdict(dict); raw=[]; efficiency=[]
    for path in sorted(Path(root).glob('*/*/evaluation.json')):
        result,run=verified_result(path.parent); raw.append(result)
        key=(result['tier'],result['variant'],result['source'],result['target'],result['label_rate'])
        # Sensitivity identities include the factor; keep them separate.
        if result['tier']=='sensitivity': key=key+(result['run_id'].split('__')[-1],)
        if result['seed'] in groups[key]: raise ValueError('Duplicate matched seed')
        groups[key][result['seed']]=result['metrics']
        efficiency.append(dict(run_id=run['run_id'],training_seconds=run['training_seconds'],
            inference_seconds=result['inference_seconds'],peak_gpu_bytes=run['peak_gpu_bytes'],
            environment=run['environment'],stages=run['checkpoint_lineage']))
    summaries=[]; complete={}; missing=[]; sensitivity_defaults=[]
    for key,rows in groups.items():
        expected=set(range(3 if key[0]=='sensitivity' else 5))
        if set(rows)!=expected:
            missing.append(dict(condition=key,missing=sorted(expected-set(rows)))); continue
        values={m:np.array([rows[s][m] for s in sorted(expected)]) for m in ('macro_f1','accuracy')}
        complete[key]=values
        if key[0:2]==('core','B6') and key[4]==.03:
            sensitivity_defaults.append(dict(condition=key[2:],seeds=[0,1,2],metrics={m:dict(mean=float(v[:3].mean()),sample_sd=float(v[:3].std(ddof=1))) for m,v in values.items()}))
        summaries.append(dict(condition=key,n=len(rows),metrics={m:dict(mean=float(v.mean()),sample_sd=float(v.std(ddof=1))) for m,v in values.items()}))
    from experiments.execution import DOMAINS
    import itertools
    conditions=[(s,t,r) for s,t in itertools.permutations(DOMAINS,2) for r in (.01,.03,.05)]
    significance=[]; negative=[]; incomplete_families=[]
    for metric in ('macro_f1','accuracy'):
        base=[complete.get(('core','B6',*c),{}).get(metric) for c in conditions]
        for family,names in [('core',CORE[:-1]),('ablation',ABLATIONS)]:
            tests=[]
            for name in names:
                comparator=[complete.get((family,name,*c),{}).get(metric) for c in conditions]
                if any(v is None for v in base+comparator): break
                tests.append(dict(family=family,metric=metric,reference='B6',comparator=name,**paired(base,comparator)))
            if len(tests)!=len(names):
                incomplete_families.append(dict(family=family,metric=metric)); continue
            for row,p in zip(tests,holm([r['raw_p'] for r in tests])): row.update(adjusted_p=p,reject=p<.05)
            significance.extend(tests)
        b0=[complete.get(('core','B0',*c),{}).get(metric) for c in conditions]
        for name in (*CORE[1:],*ABLATIONS):
            tier='core' if name in CORE else 'ablation'
            values=[complete.get((tier,name,*c),{}).get(metric) for c in conditions]
            if any(v is None for v in b0+values): continue
            delta=np.array(values)-np.array(b0)
            negative.append(dict(variant=name,metric=metric,matched_runs=90,negative_fraction=float((delta<0).mean()),
                negative_fraction_ci=bootstrap(delta,statistic='negative_fraction'),mean_ci=bootstrap(delta),
                conditions=[dict(condition=c,mean=float(v.mean()),sample_sd=float(v.std(ddof=1))) for c,v in zip(conditions,delta)]))
    report=dict(protocol=PROTOCOL,protocol_hash=digest(PROTOCOL),raw=raw,summaries=summaries,
        significance=significance,negative_transfer=negative,efficiency=efficiency,sensitivity_defaults=sensitivity_defaults,
        missing_seeds=missing,incomplete_families=incomplete_families)
    write_json(output,report)
    import csv,json
    directory=Path(output).with_suffix(''); directory.mkdir(parents=True,exist_ok=True)
    for name in ('summaries','significance','negative_transfer','efficiency','sensitivity_defaults'):
        rows=report[name]
        if not rows: continue
        with (directory/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    return report
