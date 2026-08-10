"""Loads training configs from YAML files in `configs/`.

Same pattern as the reference SigLIP script: `python train.py <config_name>` loads
`configs/<config_name>.yaml` via OmegaConf (which gives you `${...}` interpolation,
e.g. an `experiment_name` built from other fields in the same file -- see
`configs/default.yaml`).
"""
from __future__ import absolute_import, division, unicode_literals, print_function

from pathlib import Path

from omegaconf import OmegaConf

CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(config_name: str) -> dict:
    path = CONFIGS_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config `{config_name}` not found at {path}. "
            f"Available: {sorted(p.stem for p in CONFIGS_DIR.glob('*.yaml'))}"
        )
    cfg = OmegaConf.load(path)
    return cfg
