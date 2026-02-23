import json
from types import SimpleNamespace
def load_config(path):
    with open(path) as f:
        return json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
