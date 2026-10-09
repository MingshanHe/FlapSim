"""Portable replay after cloning; recorded experiment files remain untouched."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from flapsim import load_config
from flapsim.rl.env import PRESET
from .results import video


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--label',required=True,choices=['neat_seed941','neat_seed942','ppo_seed941','ppo_seed942','shared_prior'])
    p.add_argument('--group',choices=['nominal','stress'],default='nominal');p.add_argument('--video',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.video.exists():p.error('video destination exists')
    original=root/'robot_upper7_lower8.xml';digest=hashlib.sha256(original.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='flapsim_hover9_replay_') as temporary:
        stage=Path(temporary);(stage/'calibration').mkdir();(stage/'final_test').mkdir()
        for relative in ('protocol.json','phase_bank.npz','calibration/trim.json','attitude_prior.json',
                         f'final_test/{a.label}_frozen.json',f'final_test/{a.label}_{a.group}.json'):
            shutil.copyfile(root/relative,stage/relative)
        tree=ET.parse(original);assets=(load_config(PRESET).path.parent/'../meshes').resolve()
        tree.getroot().find('compiler').set('meshdir',str(assets))
        tree.write(stage/original.name,encoding='utf-8',xml_declaration=True)
        produced=video(stage,a.label,a.group);a.video.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(produced,a.video);shutil.copyfile(produced.with_suffix('.json'),a.video.with_suffix('.json'))
    assert hashlib.sha256(original.read_bytes()).hexdigest()==digest
    print(a.video.resolve())


if __name__=='__main__':main()
