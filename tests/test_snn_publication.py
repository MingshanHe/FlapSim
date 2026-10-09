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
