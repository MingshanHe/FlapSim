"""Run the finite campaign, then always materialize an honest local report."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--branch', default='research/snn-pid-v21-performance')
    args = parser.parse_args()
    args.root = args.root.resolve()
    with (args.root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        invocation = [sys.executable, '-u', '-m', 'experiments.algorithms.snn_pid_compare.development',
                      '--root', str(args.root), '--workers', str(args.workers), '--run-formal']
        with (args.root/'development.log').open('a') as log:
            child = subprocess.Popen(invocation, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            write(args.root/'supervisor.json', dict(started_unix=time.time(), child_pid=child.pid,
                                                  branch=args.branch, publication='local_only', command=invocation))
            code = child.wait()
        write(args.root/'supervisor_exit.json', dict(returncode=code, unix=time.time(),
                                                   campaign_status=read(args.root/'status.json')))
        from .reporting import generate, commit_report
        destination = generate(args.root)
        commit_report(args.root, destination, args.branch)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
