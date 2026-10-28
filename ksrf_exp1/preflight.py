"""Read-only runtime compatibility checks; never install or alter packages."""
from __future__ import annotations

import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import sys

from .common import output_dir, versions, write_json


def run(cfg):
    from packaging.requirements import Requirement
    failures = []
    requirement_path = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in requirement_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
            if installed not in requirement.specifier:
                failures.append(f"{requirement.name}: have {installed}, expected {requirement.specifier}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing package: {requirement.name}{requirement.specifier}")
    if platform.system() != "Linux":
        failures.append("Full vLLM experiment requires Linux (Google Colab GPU runtime)")
    try:
        import torch
        if not torch.cuda.is_available():
            failures.append("CUDA GPU is unavailable")
        elif not torch.cuda.is_bf16_supported():
            failures.append("Configured bfloat16 needs an Ampere-or-newer CUDA GPU (e.g. L4/A100)")
        if torch.cuda.is_available() and torch.cuda.device_count() < cfg["tensor_parallel_size"]:
            failures.append("tensor_parallel_size exceeds allocated CUDA devices")
    except ImportError:
        failures.append("Missing preinstalled Torch")
    for name in ["VLLM_PLUGINS", "VLLM_LOGITS_PROCESSOR_PLUGINS"]:
        if os.environ.get(name):
            failures.append(f"Unset {name}: unrequested plugins are forbidden")
    plugins = list(importlib.metadata.entry_points(group="vllm.logits_processors"))
    if plugins:
        failures.append("Unrequested installed vLLM logits processors: "
                        + ", ".join(plugin.name for plugin in plugins))
    try:
        # Check the native dependency required by THIS installed vLLM build;
        # never repair the environment or pin/replace Torch ourselves.
        for specification in importlib.metadata.requires("vllm") or []:
            requirement = Requirement(specification)
            if requirement.name in {"torch", "torchvision", "torchaudio"}:
                if requirement.marker and not requirement.marker.evaluate():
                    continue
                try:
                    actual = importlib.metadata.version(requirement.name)
                    if actual not in requirement.specifier:
                        failures.append(f"vLLM native dependency mismatch: {requirement.name} "
                                        f"{actual}, build requires {requirement.specifier}")
                except importlib.metadata.PackageNotFoundError:
                    failures.append(f"Missing vLLM native dependency: {specification}")
    except importlib.metadata.PackageNotFoundError:
        pass  # Already reported in the requirements check on Linux.
    if failures:
        raise RuntimeError("Environment check failed; nothing was installed or changed:\n  - "
                           + "\n  - ".join(failures))
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    if vllm.__version__ != cfg["vllm_version"]:
        raise RuntimeError(f"vLLM version must equal pinned {cfg['vllm_version']}")
    if "logits_processors" not in inspect.signature(LLM.__init__).parameters:
        raise RuntimeError("Installed vLLM does not expose the required logits-processor API")
    SamplingParams(temperature=cfg["temperature"], top_p=1., top_k=0, min_p=0.,
                   max_tokens=cfg["max_tokens"], seed=cfg["sampling_seed"], extra_args={})
    if cfg["num_keys"] != 16 or cfg["key_rank"] != 8:
        raise ValueError("Full experiment requires sixteen rank-eight keys")
    if cfg["calibration_size"] != 5000 or cfg["heldout_size"] != 1500:
        raise ValueError("Full run requires 5000 calibration / 1500 held-out prompts")
    info = {"python": sys.version, "platform": platform.platform(),
            "versions": versions(["torch", "vllm", "transformers", "sentence-transformers",
                                  "datasets", "numpy", "scipy", "scikit-learn", "huggingface-hub"]),
            "cuda": torch.version.cuda,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "installation_performed": False, "config": cfg,
            "processor_api": "vllm.v1.sample.logits_processor.AdapterLogitsProcessor"}
    write_json(output_dir(cfg) / "environment.json", info)
    print("Environment preflight: PASS (existing Torch/CUDA unchanged)", flush=True)
