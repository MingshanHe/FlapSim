"""Create a fresh portable experiment directory without touching published results."""
import argparse
from pathlib import Path
import subprocess
import sys
from .plant import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('artifacts/neat_ppo_7g8g_20261008'))
    a=p.parse_args()
    if a.root.exists() and any(a.root.iterdir()):p.error('root must be absent or empty')
    prepare(a.root,a.source)
    subprocess.run([sys.executable,'-m','experiments.algorithms.hover9.protocol','--root',str(a.root)],check=True)
    print(a.root.resolve())


if __name__=='__main__':main()
