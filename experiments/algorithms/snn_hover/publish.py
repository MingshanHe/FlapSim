"""Publish this finite, user-authorized study only after the full audit passes.

This command does not schedule recurring work, change experimental results,
force-push, or publish a partial report. State is kept outside the study manifest.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import write


BRANCH = 'research/neat-snn-hover'
REMOTE = 'https://github.com/MingshanHe/FlapSim.git'


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def windows_git():
    executable = Path('/mnt/e/功能软件/Git/cmd/git.exe')
    if not executable.is_file():
        raise RuntimeError('Configured Windows Git is unavailable')
    checkout = Path.cwd().resolve()
    gitdir = Path(git('rev-parse', '--absolute-git-dir')).resolve()
    def unc(path):
        return '\\\\wsl.localhost\\Ubuntu' + str(path).replace('/', '\\')
    return [str(executable), '--git-dir=' + unc(gitdir), '--work-tree=' + unc(checkout)]


def remote_head(command):
    result = subprocess.check_output(command + ['ls-remote', REMOTE, 'refs/heads/' + BRANCH], text=True, timeout=120)
    rows = [line.split() for line in result.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][1] != 'refs/heads/' + BRANCH:
        raise RuntimeError('Expected remote research branch was not found')
    return rows[0][0]


def check_scope(root):
    allowed = [str(root), 'experiments/algorithms/snn_hover', 'tests/test_snn_hover.py',
               'tests/test_snn_analysis.py', 'tests/test_snn_publication.py', 'report.md', 'report_final.md']
    def in_scope(path):
        return any(path == prefix or path.startswith(prefix + '/') for prefix in allowed)
    if git('branch', '--show-current') != BRANCH:
        raise RuntimeError('Worktree branch changed; refusing publication')
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Index already contains staged work; refusing to alter it')
    unexpected = [path for path in git('diff', '--name-only').splitlines() if not in_scope(path)]
    if unexpected:
        raise RuntimeError('Tracked changes outside authorized study scope: ' + ', '.join(unexpected))
    return allowed


def check_ready(root):
    def read(name):
        return json.loads((root / name).read_text())
    if read('finish_status.json').get('status') != 'completed':
        raise RuntimeError('Full training, independent testing, media and report are not complete')
    if read('campaign_status.json').get('completed') != 35:
        raise RuntimeError('Expected all 35 independent training runs')
    if read('evaluation_status.json').get('completed') != 11888:
        raise RuntimeError('Expected all 11,888 independent evaluation episodes')
    if read('verification.json').get('status') != 'passed':
        raise RuntimeError('Final audit did not pass')
    if not Path('report_final.md').is_file():
        raise RuntimeError('Standalone final report is missing')
    for path in root.rglob('*'):
        if path.is_file() and path.stat().st_size >= 95 * 1024 * 1024:
            raise RuntimeError('Artifact exceeds conservative GitHub file-size limit: ' + str(path))


def publish(root, state):
    allowed = check_scope(root)
    check_ready(root)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    # The study is now idle. This second audit seals logs after finish.py closed
    # them, avoiding a manifest entry for a still-being-written verification log.
    subprocess.run([sys.executable, '-m', 'experiments.algorithms.snn_hover.verify', '--root', str(root)],
                   check=True, env=env, timeout=3600)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_snn_hover.py',
                    'tests/test_snn_analysis.py', 'tests/test_snn_publication.py', 'tests/test_topology_hover.py'],
                   check=True, env=env, timeout=600)
    command = windows_git()
    before = git('rev-parse', 'HEAD')
    remote_before = remote_head(command)
    subprocess.run(['git', 'merge-base', '--is-ancestor', remote_before, before], check=True)
    check_scope(root)
    subprocess.run(['git', 'diff', '--check'], check=True)
    subprocess.run(['git', 'add', '--', *allowed], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    staged = git('diff', '--cached', '--name-only').splitlines()
    if staged:
        subprocess.run(['git', 'commit', '-m', 'Complete NEAT-SNN hover study with final research results and reproducible evidence'], check=True)
    commit = git('rev-parse', 'HEAD')
    write(state, dict(status='pushing',branch=BRANCH,commit=commit,previous_remote=remote_before,
                     final_report='report_final.md',time=time.time()))
    subprocess.run(command + ['push', REMOTE, 'HEAD:refs/heads/' + BRANCH], check=True, timeout=1800)
    if remote_head(command) != commit:
        raise RuntimeError('Remote branch does not match the verified local commit')
    result = dict(status='pushed',branch=BRANCH,commit=commit,remote=REMOTE,
                  final_report='report_final.md',training_runs=35,test_episodes=11888,time=time.time())
    write(state,result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    root = args.root
    if root.is_absolute() or '..' in root.parts:
        raise ValueError('Use the repository-relative study root')
    check_scope(root)
    command = windows_git()
    if args.preflight:
        print(json.dumps(dict(status='read_only_preflight_passed',branch=BRANCH,
                              remote_commit=remote_head(command),checkout=str(Path.cwd()))))
        return
    state = root.parent / (root.name + '_publication_status.json')
    started = time.monotonic()
    try:
        if args.wait:
            write(state,dict(status='waiting_for_complete_study',branch=BRANCH,time=time.time()))
            while True:
                for name in ['campaign_failure.json','finish_failure.json']:
                    if (root / name).exists():
                        raise RuntimeError('Study stopped; partial results will not be published as final: ' + name)
                finish = root / 'finish_status.json'
                if finish.exists() and json.loads(finish.read_text()).get('status') == 'completed':
                    break
                if time.monotonic() - started > 7 * 24 * 3600:
                    raise TimeoutError('Finite seven-day completion bound reached')
                time.sleep(30)
        publish(root,state)
    except BaseException as error:
        write(state,dict(status='failed',error=repr(error),branch=BRANCH,time=time.time()))
        raise


if __name__ == '__main__':
    main()
