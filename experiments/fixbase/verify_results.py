"""Verify the executed notebook, recorded arrays, and encoded video."""
from pathlib import Path
import json
import numpy as np
import nbformat
import imageio.v2 as imageio
from PIL import Image

folder = Path(__file__).resolve().parent
output = folder / 'results'
summary = json.loads((output / 'summary.json').read_text())
with np.load(output / 'measurements.npz') as archive:
    data = {k: archive[k] for k in archive.files}
csv = np.genfromtxt(output / 'measurements.csv', names=True, delimiter=',')
config = summary['experiment']
expected_samples = round(config['duration'] * config['sample_hz']) + 1
assert len(csv) == summary['samples'] == expected_samples
for key in data:
    assert np.isfinite(data[key]).all()
    np.testing.assert_allclose(csv[key], data[key], atol=1e-12)
for axis in ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'):
    unit = 'N' if axis.startswith('F') else 'Nm'
    np.testing.assert_allclose(data[f'aero_{axis}_{unit}'],
                              data[f'left_aero_{axis}_{unit}'] + data[f'right_aero_{axis}_{unit}'], atol=1e-12)
    np.testing.assert_allclose(data[f'fixture_{axis}_{unit}'], -data[f'support_{axis}_{unit}'])
assert not np.any(data['base_pose_error'])
assert min(summary['pitch_reversals_after_025s']) >= 4

nb = nbformat.read(folder / 'IST_robot_fixbase.ipynb', as_version=4)
nbformat.validate(nb)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
assert any('video/mp4' in str(o) for c in nb.cells if c.cell_type == 'code' for o in c.outputs)

reader = imageio.get_reader(output / 'fixbase.mp4')
expected_frames = (expected_samples - 1) // (config['sample_hz'] // config['video_sample_hz']) + 1
assert reader.count_frames() == summary['video_frames'] == expected_frames
assert reader.get_meta_data()['fps'] == config['playback_fps']
frames = [np.asarray(reader.get_data(i)) for i in (20, 40, 60, 80)]
reader.close()
assert all(f.shape == (560, 1280, 3) for f in frames)
assert np.mean(np.abs(frames[0].astype(float)-frames[2])) > 2
contact = Image.new('RGB', (1280, 1120), 'white')
for i, frame in enumerate(frames):
    contact.paste(Image.fromarray(frame).resize((640, 280)), ((i % 2)*640, (i // 2)*560))
    # Lower part is the same full frame's wing closeup, enlarged for inspection.
    crop = Image.fromarray(frame).crop((640, 80, 1280, 560)).resize((640, 280))
    contact.paste(crop, ((i % 2)*640, (i // 2)*560+280))
contact.save(output / 'video_contact_sheet.jpg', quality=90)
print(f'Verified: {len(csv)} finite samples, six-axis sums/signs, executed notebook, {expected_frames} decoded video frames.')
