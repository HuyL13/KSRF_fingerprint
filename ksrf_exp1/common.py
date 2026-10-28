"""Small artifact boundary shared by independent pipeline stages."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from ._reused.imf_data_io import _atomic_text, read_jsonl as _read_jsonl
from ._reused.imf_provenance import sha256_file as file_checksum


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return _read_jsonl(Path(path))


def write_json(path, value):
    _atomic_text(Path(path), json.dumps(value, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n")


def write_jsonl(path, records):
    # Serialize before opening a temporary file: invalid numeric values leave
    # both the previous artifact and its directory untouched.
    content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                      for row in records)
    _atomic_text(Path(path), content)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        array = np.asarray(array)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite artifact: {name}")
    np.savez_compressed(path, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def versions(names):
    return {name: importlib.metadata.version(name) for name in names}


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["outputs"] = str(Path(cfg["outputs"]).resolve())
    return cfg


def output_dir(cfg):
    return Path(cfg["outputs"])


def require_finite(name, value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite values in {name}")
    return value
