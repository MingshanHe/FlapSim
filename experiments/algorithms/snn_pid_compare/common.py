"""Atomic experiment records and immutable input provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import threading
from contextlib import contextmanager

_STATUS_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parents[3]
DT = .002


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def event(root, stage, **details):
    record = dict(stage=stage, unix=time.time(), pid=os.getpid(), **details)
    with _STATUS_LOCK:
        write(Path(root)/'status.json', record)
    append(Path(root)/'events.jsonl', record)
    print(json.dumps(record), flush=True)


@contextmanager
def heartbeat(root):
    """Refresh visible progress without allowing a stale phase to overwrite events."""
    root = Path(root)
    stopped = threading.Event()
    def update():
        previous_cpu = None
        while not stopped.wait(10):
            with _STATUS_LOCK:
                if not (root/'status.json').exists():
                    continue
                record = read(root/'status.json')
                record.update(heartbeat_unix=time.time(), heartbeat_pid=os.getpid())
                cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
                idle, total = cpu[3]+cpu[4], sum(cpu)
                if previous_cpu is not None and total > previous_cpu[1]:
                    record['host_cpu_percent'] = round(100*(1-(idle-previous_cpu[0])/(total-previous_cpu[1])), 1)
                previous_cpu = idle, total
                record['load_average'] = list(os.getloadavg())
                progress = {}
                for pattern in ('development/round*/*/status.json', 'initialization/*/round*/status.json', 'training/*/status.json'):
                    for path in root.glob(pattern):
                        child = read(path)
                        progress[str(path.parent.relative_to(root))] = {
                            key: child[key] for key in ('epoch', 'generation', 'update', 'loss', 'training_steps', 'budget') if key in child}
                        progress[str(path.parent.relative_to(root))]['completed'] = any((path.parent/name).exists() for name in ('completed.json', 'results.json'))
                record['model_progress'] = progress
                write(root/'status.json', record)
    thread = threading.Thread(target=update, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def prepare_source(destination):
    """Only relocate meshdir in a new copy; never edit the frozen source."""
    import re
    import shutil
    import xml.etree.ElementTree as ET
    source = ROOT/'artifacts/pid_airborne_20261008'
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    originals = {}
    for relative in ('calibration/trim.json', 'phase_bank.npz', 'selected_pid.json'):
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, target)
        originals[relative] = sha(source/relative)
    original = source/'robot_upper7_lower8.xml'
    text = original.read_text()
    meshdir = str(ROOT/'flapsim/robot/IST_robot-V2/meshes')
    adjusted, count = re.subn(r'meshdir="[^"]+"', 'meshdir="'+meshdir+'"', text)
    if count != 1:
        raise ValueError('Expected exactly one meshdir')
    before, after = ET.fromstring(text), ET.fromstring(adjusted)
    before.find('compiler').set('meshdir', meshdir)
    assert ET.tostring(before) == ET.tostring(after)
    target = destination/original.name
    target.write_text(adjusted)
    originals[original.name] = sha(original)
    write(destination/'provenance.json', dict(source=str(source), original_sha256=originals,
          relocated_xml_sha256=sha(target), only_change='compiler.meshdir', meshdir=meshdir))
    return destination
