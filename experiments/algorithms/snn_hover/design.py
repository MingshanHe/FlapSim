"""Read immutable training protocol plus a pre-test evaluation-only addendum."""
import json
from pathlib import Path

def read_protocol(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    addendum=root/'evaluation_addendum.json'
    if addendum.exists():
        a=json.loads(addendum.read_text());assert a['status']=='frozen_before_independent_tests'
        p['factors']['names']=a['factor_names'];p['evaluation_addendum']=a
    return p
