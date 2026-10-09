"""Seal a completed-run handoff and verify it before explicit continuation."""
import argparse,hashlib,importlib.metadata,json,platform,shutil,subprocess,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
from .common import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def seal(root):
    pause=read(root/'pause_completed.json')
    assert pause['status']=='paused' and pause['after_run']=='snn_neat_seed2202'
    completed=sorted(p.parent for p in (root/'training').glob('*/results.json') if read(p)['status']=='completed')
    assert len(completed)==8 and not (root/'evaluation/episodes.jsonl').exists()
    p=read(root/'protocol.json')
    assert {x.name for x in completed}=={f'{m}_seed2201' for m in p['methods']}|{'snn_neat_seed2202'}
    for folder in completed:
        result=read(folder/'results.json');assert result['budget']==p['final_budget']
        assert len(list(folder.glob('validation_*.json')))==21
        for budget in [256000,512000,1024000]:assert (folder/f'budget_{budget}/best_policy.json').exists()
    files={Path(x) for x in p['source_hashes']}
    files.update(Path(x) for x in read(root/'evaluation_code_hashes.json')['files'])
    files.update(x for x in (root/'frozen_source').rglob('*') if x.is_file())
    tracked=subprocess.check_output(['git','ls-files','flapsim','experiments/algorithms/neat_imu'],text=True).splitlines()
    files.update(Path(x) for x in tracked)
    for name in ['robot_upper7_lower8.xml','phase_bank.npz','calibration/trim.json','selected_pid.json']:
        files.add(Path('artifacts/pid_airborne_20261008')/name)
    for method in ['neat','fixed_evo','rnn_ppo','ff_ppo']:
        for seed in range(1201,1206):
            files.update(Path(f'artifacts/topology_hover_20261008/pilot/{method}_seed{seed}')/name for name in ['best_policy.json','selection.json','results.json'])
    evidence=Path('docs/research/snn_neat_20261009/evidence')
    files.update(x for x in evidence.iterdir() if x.is_file())
    for folder in completed:
        files.update(x for x in folder.rglob('*') if x.is_file() and 'visuals' not in x.relative_to(folder).parts)
    files.update(root/name for name in ['protocol.json','protocol_revision.json','evaluation_addendum.json','evaluation_code_hashes.json','pause_request.json','pause_completed.json'])
    manifest=dict(status='sealed_completed_run_boundary',created_unix=time.time(),completed_runs=[x.name for x in completed],
        remaining_runs=27,next_run='snn_fixed_tau_seed2202',base_commit=p['base_commit'],
        repository_head_at_sealing=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        origin_host=platform.node(),origin_platform=platform.platform(),
        note='Hashes are authoritative for migration. Visuals may be regenerated; no neural/physical state needs mid-episode restoration.',
        files={str(x):dict(sha256=sha(x),bytes=x.stat().st_size) for x in sorted(files)})
    write(root/'migration_manifest.json',manifest)
    return dict(status='sealed',files=len(files),completed_runs=len(completed))


def check(root,replay=False,allow_unsealed=False,render=False):
    p=read(root/'protocol.json')
    subprocess.run(['git','cat-file','-e',p['base_commit']+'^{commit}'],check=True,capture_output=True)
    for path,expected in p['source_hashes'].items():assert sha(path)==expected,path
    for path,expected in read(root/'evaluation_code_hashes.json')['files'].items():assert sha(path)==expected,path
    manifest=root/'migration_manifest.json'
    if manifest.exists():
        m=read(manifest)
        for path,record in m['files'].items():assert sha(path)==record['sha256'],f'Migration file changed: {path}'
    elif not allow_unsealed:raise RuntimeError('No sealed migration manifest. Wait for the requested run boundary.')
    versions={name:importlib.metadata.version(name) for name in p['packages']}
    for name,expected in p['packages'].items():assert versions[name].split('+')[0]==expected,(name,versions[name],expected)
    assert platform.system()=='Linux','Use Linux or Ubuntu under WSL2; the frozen native backend requires Linux .so libraries'
    assert platform.python_version()==p['python'],(platform.python_version(),p['python'])
    assert shutil.which('g++'),'g++ is required to rebuild native dynamics/sensor modules'
    xml=Path('artifacts/pid_airborne_20261008/robot_upper7_lower8.xml');tree=ET.parse(xml)
    meshdir=Path(tree.getroot().find('compiler').attrib['meshdir'])
    expected_meshdir=Path.cwd()/'flapsim/robot/IST_robot-V2/meshes'
    assert meshdir.resolve()==expected_meshdir.resolve(),'Canonical mesh path points to another checkout; follow the verified symlink step in exec.md'
    meshes=[meshdir / x.attrib['file'] for x in tree.findall('.//mesh')]
    assert all(x.is_file() for x in meshes),'Frozen absolute mesh path unavailable. Create the verified canonical repository symlink in exec.md; do not edit the XML.'
    result=dict(status='passed',host=platform.node(),platform=platform.platform(),python=platform.python_version(),packages=versions,
        frozen_core_files=len(p['source_hashes']),migration_files=len(m['files']) if manifest.exists() else None,
        source_checkout=str(Path.cwd()),mesh_files=len(meshes),formal_test_data_accessed=False,validation_replay_steps=0)
    if replay:
        from .common import evaluate
        import numpy as np
        folder=root/'training/snn_neat_seed2201';selection=read(folder/'selection.json')
        expected=read(folder/f'validation_{selection["allocated_steps"]:07d}.json')[:2]
        spec=read(folder/'best_policy.json');records=[]
        for row in expected:
            actual=evaluate((spec,row['case'],'residual','height',False))
            keys=['return_sum','tilt_rms_deg','height_rmse_m','minimum_clearance_m','elapsed']
            delta=max(abs(actual[k]-row[k]) for k in keys)
            np.testing.assert_allclose([actual[k] for k in keys],[row[k] for k in keys],rtol=0,atol=1e-6)
            assert actual['success']==row['success'] and actual['termination']==row['termination']
            records.append(dict(seed=row['seed'],condition=row['case']['condition'],maximum_metric_absolute_error=delta,steps=actual['steps']))
        result.update(validation_replays=records,validation_replay_steps=sum(x['steps'] for x in records),
            replay_scope='Two already-used seed2201 validation cases, not independent tests or extra training.')
    if render:
        import mujoco
        import numpy as np
        import imageio_ffmpeg
        from PIL import ImageFont
        model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        renderer=mujoco.Renderer(model,height=64,width=64)
        try:
            renderer.update_scene(data);frame=renderer.render()
            assert frame.shape==(64,64,3) and np.isfinite(frame).all() and np.ptp(frame)>0
        finally:renderer.close()
        ImageFont.truetype('DejaVuSans.ttf',18)
        encoder=subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(),'-version'],text=True).splitlines()[0]
        result['render_check']=dict(shape=list(frame.shape),pixel_range=int(np.ptp(frame)),encoder=encoder,
            scope='Static model frame, font and encoder; zero physical simulation steps')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seal',action='store_true');p.add_argument('--replay-check',action='store_true')
    p.add_argument('--render-check',action='store_true')
    p.add_argument('--allow-unsealed',action='store_true',help='Development-only check before the active run finishes')
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.output and a.output.exists():raise FileExistsError('Keep prior machine checks; choose a new output filename')
    result=seal(a.root) if a.seal else check(a.root,a.replay_check,a.allow_unsealed,a.render_check)
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
