"""Explicit compatibility audit, separate from the experiment runner."""
from __future__ import annotations

import importlib.metadata
import os

from .common import output_dir, versions, write_json
from .doctor import audit_environment


def run(cfg):
    errors = []
    if cfg["num_keys"] != 16 or cfg["key_rank"] != 8:
        errors.append("Full experiment requires sixteen rank-eight keys")
    if cfg["calibration_size"] != 5000 or cfg["heldout_size"] != 1500:
        errors.append("Full run requires 5000 calibration / 1500 held-out prompts")
    for name in ["VLLM_PLUGINS", "VLLM_LOGITS_PROCESSOR_PLUGINS"]:
        if os.environ.get(name):
            errors.append(f"Unset {name}: unrequested plugins are forbidden")
    plugins = list(importlib.metadata.entry_points(group="vllm.logits_processors"))
    if plugins:
        errors.append("Unrequested installed vLLM logits processors: "
                      + ", ".join(plugin.name for plugin in plugins))
    if errors:
        raise RuntimeError("Preflight failed:\n  - " + "\n  - ".join(errors))
    report = audit_environment()
    native = report["native"]["details"]
    if len(native.get("devices", [])) < cfg["tensor_parallel_size"]:
        report["errors"].append("tensor_parallel_size exceeds allocated CUDA devices")
        report["status"] = "FAIL"
    report.update(config=cfg, cuda=native.get("torch_cuda_build"),
                  gpu_names=[device["name"] for device in native.get("devices", [])],
                  processor_api="vllm.v1.sample.logits_processor.AdapterLogitsProcessor")
    if report["status"] == "PASS":
        report["versions"] = versions(["torch", "vllm", "transformers", "sentence-transformers",
                                       "datasets", "numpy", "scipy", "scikit-learn", "huggingface-hub"])
    write_json(output_dir(cfg) / "environment.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("Environment/import/native-CUDA check failed; nothing was installed or changed:\n  - "
                           + "\n  - ".join(report["errors"]))
    print(f"Environment preflight: PASS ({len(report['imports'])} actual import checks; "
          "vLLM CUDA kernel executed; existing Torch/CUDA unchanged)", flush=True)
