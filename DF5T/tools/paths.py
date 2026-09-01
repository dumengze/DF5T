from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', project_root()))
    return project_root()


def resource_path(relative_path: str) -> str:
    return str((runtime_root() / relative_path).resolve())


def ensure_dir(path: os.PathLike | str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(Path(path))


def existing_first(candidates: Iterable[os.PathLike | str | None]) -> Optional[str]:
    for item in candidates:
        if not item:
            continue
        p = Path(item).expanduser()
        if p.is_file() or p.is_dir():
            return str(p.resolve())
    return None


def default_model_candidates(name: str = 'model_2562.pt') -> List[str]:
    root = project_root()
    return [
        str(root / name),
        str(root / 'weights' / name),
        str(root / 'checkpoints' / name),
        str(root / 'models' / name),
        str(root / 'pretrained' / name),
        str(root / 'exp' / 'model' / 'MitEM' / 'model_y.pt'),
        str(Path.cwd() / name),
    ]


def resolve_model_path(preferred: Optional[str] = None, env_var: str = 'DF5T_MODEL_PATH') -> Optional[str]:
    env_value = os.environ.get(env_var)
    return existing_first([preferred, env_value, *default_model_candidates()])


def resolve_config_path(preferred: Optional[str] = None) -> Optional[str]:
    root = project_root()
    return existing_first([
        preferred,
        root / 'configs' / 'DF5T_256.yml',
        root / 'configs' / 'DF5T_512.yml',
        runtime_root() / 'configs' / 'DF5T_256.yml',
        runtime_root() / 'configs' / 'DF5T_512.yml',
    ])


def resolve_data_root(preferred: Optional[str] = None) -> Optional[str]:
    root = project_root()
    return existing_first([
        preferred,
        root / 'datasets' / 'MitEM' / 'MitEM',
        root / 'data',
        Path.cwd() / 'datasets' / 'MitEM' / 'MitEM',
        Path.cwd() / 'data',
    ])
