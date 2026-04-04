"""Load config.yaml and expose a Config object with nested attribute access."""

import types
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _to_namespace(obj):
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(path: Path = _CONFIG_PATH) -> types.SimpleNamespace:
    with open(path) as f:
        data = yaml.safe_load(f)
    return _to_namespace(data)


CFG = load_config()
