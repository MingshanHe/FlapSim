"""Publication must not turn partial experiments or unrelated edits into a push."""
import json
from pathlib import Path
import pytest
from experiments.algorithms.snn_hover import publish


def ready_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=Path('artifacts/study');root.mkdir(parents=True)
    values={'finish_status.json':dict(status='completed'),
            'campaign_status.json':dict(completed=35),
            'evaluation_status.json':dict(completed=11888),
            'verification.json':dict(status='passed')}
    for name,value in values.items():(root/name).write_text(json.dumps(value))
    Path('report_final.md').write_text('Synthetic test fixture only')
    return root


@pytest.mark.parametrize('name,value',[
    ('finish_status.json',dict(status='running')),
    ('campaign_status.json',dict(completed=34)),
    ('evaluation_status.json',dict(completed=11887)),
    ('verification.json',dict(status='failed'))])
def test_incomplete_or_failed_evidence_blocks_publication(tmp_path,monkeypatch,name,value):
    root=ready_fixture(tmp_path,monkeypatch)
    (root/name).write_text(json.dumps(value))
    with pytest.raises(RuntimeError):publish.check_ready(root)


def test_missing_standalone_report_blocks_publication(tmp_path,monkeypatch):
    root=ready_fixture(tmp_path,monkeypatch)
    Path('report_final.md').unlink()
    with pytest.raises(RuntimeError,match='report is missing'):publish.check_ready(root)


@pytest.mark.parametrize('branch,staged,changed',[
    ('v3','',''),
    (publish.BRANCH,'user_file.py',''),
    (publish.BRANCH,'','experiments/algorithms/topology_hover/environment.py')])
def test_unrelated_work_is_not_committed(monkeypatch,branch,staged,changed):
    values={('branch','--show-current'):branch,('diff','--cached','--name-only'):staged,
            ('diff','--name-only'):changed}
    monkeypatch.setattr(publish,'git',lambda *args:values[args])
    with pytest.raises(RuntimeError):publish.check_scope(Path('artifacts/study'))


def test_complete_evidence_passes_readiness_guard(tmp_path,monkeypatch):
    root=ready_fixture(tmp_path,monkeypatch)
    publish.check_ready(root)


def test_user_pause_blocks_even_otherwise_complete_publication(tmp_path,monkeypatch):
    root=ready_fixture(tmp_path,monkeypatch)
    (root/'pause_request.json').write_text(json.dumps(dict(after_run='snn_neat_seed2202')))
    with pytest.raises(RuntimeError,match='paused by user'):publish.check_ready(root)


def test_resume_skips_completed_run_and_preserves_pause_record(tmp_path,monkeypatch):
    from experiments.algorithms.snn_hover import campaign
    root=tmp_path/'study';root.mkdir();training=root/'training';training.mkdir()
    protocol=dict(methods=['snn_neat','snn_fixed_tau'],seeds=[2202],final_budget=1,
                  workers=16,population=64,source_hashes={})
    (root/'protocol.json').write_text(json.dumps(protocol))
    (root/'pause_request.json').write_text(json.dumps(dict(after_run='snn_neat_seed2202')))
    (root/'pause_completed.json').write_text(json.dumps(dict(status='paused',after_run='snn_neat_seed2202')))
    def completed(folder,method):
        folder.mkdir()
        (folder/'results.json').write_text(json.dumps(dict(status='completed',budget=1,method=method,seed=2202,
            training_steps=1,best_validation=dict(success_rate=1.,tilt_rms_deg=1.))))
    completed(training/'snn_neat_seed2202','snn_neat')
    original=(training/'snn_neat_seed2202/results.json').read_bytes();started=[]
    class FakeProcess:
        pid=999999
        returncode=0
        def __init__(self,command,**kwargs):
            # No actual trainer is executed by this orchestration test.
            folder=Path(command[command.index('--output')+1]);started.append(folder.name)
            completed(folder,'snn_fixed_tau')
        def poll(self):return 0
    monkeypatch.setattr(campaign.platform,'platform',lambda:'synthetic-test-platform')
    monkeypatch.setattr(campaign.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(campaign.sys,'argv',['campaign','--root',str(root),'--resume'])
    campaign.main()
    assert started==['snn_fixed_tau_seed2202']
    assert (training/'snn_neat_seed2202/results.json').read_bytes()==original
    assert not (root/'pause_request.json').exists()
    archived=list((root/'control').glob('resume_*.json'))
    assert len(archived)==1 and json.loads(archived[0].read_text())['pause_request']['after_run']=='snn_neat_seed2202'
