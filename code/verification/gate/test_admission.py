import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.cache.store import file_hash, write_json
from verification.gate import run


@pytest.mark.parametrize('mutation',['failed','skipped','missing','stale','evidence'])
def test_admission_fails_closed(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(run,'snapshot',lambda:'current')
    evidence=tmp_path/'evidence.txt'
    evidence.write_text('verified')
    report={'status':'PASS','code_snapshot_hash':'current',
        **{k:{'status':'PASS'} for k in run.REQUIRED},
        'checks':[{'name':k,'status':'PASS'} for k in list(run.GROUPS)+['synthetic_smoke','full_chain_smoke']],
        'evidence_hashes':{str(evidence):file_hash(evidence)}}
    path=tmp_path/'report.json'
    write_json(path,report)
    run.require_pass(path)
    if mutation=='failed': report['status']='FAIL'
    if mutation=='skipped': report['checks'][0]['status']='SKIPPED'
    if mutation=='missing': del report[run.REQUIRED[0]]
    if mutation=='stale': report['code_snapshot_hash']='previous'
    if mutation=='evidence': evidence.write_text('changed')
    write_json(path,report)
    with pytest.raises(PermissionError): run.require_pass(path)


def test_non_smoke_cli_requires_gate():
    from types import SimpleNamespace
    from experiments.scheduling.admission import check_cli
    with pytest.raises(PermissionError): check_cli(SimpleNamespace(smoke=False))
    assert check_cli(SimpleNamespace(smoke=True)) is None


def test_project_interpreter_is_platform_independent(tmp_path):
    venv = tmp_path / '.venv'
    run.require_project_interpreter(prefix=str(venv), base_prefix=str(tmp_path / 'base'), repo=tmp_path)
    with pytest.raises(RuntimeError):
        run.require_project_interpreter(prefix=str(tmp_path / 'other'), base_prefix=str(tmp_path / 'base'), repo=tmp_path)
    with pytest.raises(RuntimeError):
        run.require_project_interpreter(prefix=str(venv), base_prefix=str(venv), repo=tmp_path)
