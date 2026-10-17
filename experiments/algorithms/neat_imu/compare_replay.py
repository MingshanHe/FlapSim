"""Replay saved comparison policies after cloning into a different directory.

Only the mesh search directory is relocated in a temporary XML copy. The
recorded model and its hash remain untouched, and the flight result is checked.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from flapsim import load_config
from flapsim.rl.env import PRESET
from .compare_results import record_video


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--group',choices=('nominal','stress'),default='nominal')
    parser.add_argument('--video',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve()
    if args.label not in ('neat_seed835','neat_seed836','ppo_seed835','ppo_seed836'):
        parser.error('label must identify one of the four frozen experiment policies')
    if args.video.exists(): parser.error('video path already exists')
    model=root/'robot_upper7_lower8.xml'; original_hash=hashlib.sha256(model.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='flapsim_imu6_replay_') as temporary:
        stage=Path(temporary); (stage/'calibration').mkdir(); (stage/'final_test').mkdir()
        for relative in ('protocol.json','phase_bank.npz','calibration/trim.json',
                         f'final_test/{args.label}_frozen.json',f'final_test/{args.label}_{args.group}.json'):
            shutil.copyfile(root/relative,stage/relative)
        tree=ET.parse(model)
        assets=(load_config(PRESET).path.parent/'../meshes').resolve()
        tree.getroot().find('compiler').set('meshdir',str(assets))
        tree.write(stage/model.name,encoding='utf-8',xml_declaration=True)
        produced=Path(record_video(stage,args.label,args.group))
        args.video.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(produced,args.video)
        shutil.copyfile(produced.with_suffix('.json'),args.video.with_suffix('.json'))
    assert hashlib.sha256(model.read_bytes()).hexdigest()==original_hash
    print(str(args.video.resolve()))


if __name__=='__main__': main()
