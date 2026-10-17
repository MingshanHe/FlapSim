"""Reject tampered frozen artifacts before deserializing policy or normalizer."""
import hashlib
import json

import pytest

from flapsim.rl.train import load_run


@pytest.mark.parametrize('filename,key', [
    ('policy.zip', 'exported_policy_sha256'),
    ('vecnormalize.pkl', 'exported_normalizer_sha256'),
])
def test_modified_export_rejected_before_loading(tmp_path, filename, key):
    expected = hashlib.sha256(b'original').hexdigest()
    (tmp_path/'run.json').write_text(json.dumps({key: expected}))
    (tmp_path/filename).write_bytes(b'changed')
    with pytest.raises(ValueError, match='exported checkpoint changed'):
        load_run(tmp_path)
