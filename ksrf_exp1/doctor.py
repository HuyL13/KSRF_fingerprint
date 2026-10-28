"""Read-only version, real-import and CUDA-extension audit; no installation.

Run independently with ``python -m ksrf_exp1.doctor``. Only the standard
library is imported at startup, so missing ML dependencies can be reported.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys


def collect_imports(package_root):
    root = Path(package_root)
    found = {}
    for path in sorted(root.rglob("*.py")):
        package = ".".join((root.name, *path.relative_to(root).parts[:-1]))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            entries = []
            if isinstance(node, ast.Import):
                entries = [(alias.name, None) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                module = "." * node.level + (node.module or "")
                if node.level:
                    module = importlib.util.resolve_name(module, package)
                entries = [(module, alias.name) for alias in node.names]
            for module, symbol in entries:
                key = (module, symbol)
                found.setdefault(key, {"module": module, "symbol": symbol, "locations": []})
                found[key]["locations"].append(f"{path.relative_to(root)}:{node.lineno}")
    return list(found.values())


def check_imports(entries):
    results = []
    for entry in entries:
        row = dict(entry)
        try:
            module = importlib.import_module(entry["module"])
            if entry["symbol"] and entry["symbol"] != "*":
                getattr(module, entry["symbol"])
            row["status"] = "PASS"
        except Exception as error:
            row.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        results.append(row)
    return results


def validate_dependency_versions(requirements, installed, environment=None):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    installed = {canonicalize_name(name): version for name, version in installed.items()}
    errors = []
    for specification in requirements:
        requirement = Requirement(specification)
        marker_environment = dict(environment or {}, extra=(environment or {}).get("extra", ""))
        if requirement.marker and not requirement.marker.evaluate(marker_environment):
            continue
        version = installed.get(canonicalize_name(requirement.name))
        if version is None:
            errors.append(f"Missing package: {requirement.name}{requirement.specifier}")
        elif version not in requirement.specifier:
            errors.append(f"{requirement.name}: have {version}, required {requirement.specifier}")
    return errors


def dependency_audit(requirements):
    """Inspect mandatory dependency metadata recursively, including active extras."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    installed = {canonicalize_name(dist.metadata["Name"]): dist.version
                 for dist in metadata.distributions() if dist.metadata["Name"]}
    errors = validate_dependency_versions(requirements, installed)
    queue = [Requirement(spec) for spec in requirements]
    seen = set()
    checked = {}
    while queue:
        requirement = queue.pop()
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        key = (name, tuple(sorted(requirement.extras)))
        if key in seen or name not in installed:
            continue
        seen.add(key)
        checked[name] = installed[name]
        for specification in metadata.requires(name) or []:
            child = Requirement(specification)
            active = any(child.marker is None or child.marker.evaluate({"extra": extra})
                         for extra in ({"", *requirement.extras}))
            if not active:
                continue
            unmarked = str(child).split(";", 1)[0]
            for error in validate_dependency_versions([unmarked], installed):
                errors.append(f"{name} requires {unmarked}: {error}")
            queue.append(Requirement(unmarked))
    return {"checked_versions": checked, "errors": sorted(set(errors))}


def gpu_and_native_audit(expected_vllm):
    details = {}
    errors = []
    if platform.system() != "Linux":
        errors.append("Full vLLM experiment requires Linux (Google Colab GPU runtime)")
    try:
        import torch
        details.update(torch=torch.__version__, torch_cuda_build=torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable")
        devices = [{"name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i))}
                   for i in range(torch.cuda.device_count())]
        details["devices"] = devices
        if any(device["capability"][0] < 8 for device in devices):
            raise RuntimeError("bfloat16 vLLM requires compute capability >= 8.0 (e.g. L4/A100); T4 is unsupported")
        installed_vllm = metadata.version("vllm")
        details["vllm"] = installed_vllm
        if installed_vllm != expected_vllm:
            raise RuntimeError(f"Expected pinned vLLM {expected_vllm}, found {installed_vllm}")
        # Matching torch's public version is necessary but not sufficient for ABI
        # compatibility. Load and EXECUTE the installed native CUDA extension.
        importlib.import_module("vllm._C")
        from vllm import _custom_ops
        x = torch.arange(128, device="cuda", dtype=torch.float32).reshape(4, 32) / 128
        x = x.to(torch.bfloat16)
        weight = torch.ones(32, device="cuda", dtype=x.dtype)
        out = torch.empty_like(x)
        _custom_ops.rms_norm(out, x, weight, 1e-6)
        expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(out.float(), expected, rtol=.02, atol=.02)
        torch.cuda.synchronize()
        details["vllm_cuda_rms_norm"] = "PASS"
    except Exception as error:
        errors.append(f"CUDA/native extension: {type(error).__name__}: {error}")
        details["vllm_cuda_rms_norm"] = "NOT_PASSED"
    return {"details": details, "errors": errors}


def api_audit():
    from vllm import LLM, SamplingParams
    from vllm.config import ModelConfig
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    from sentence_transformers import SentenceTransformer
    from scipy.sparse.linalg import cg
    for function, names in [(LLM.__init__, ["logits_processors", "revision", "tokenizer_revision"]),
                            (SentenceTransformer.__init__, ["revision", "device", "trust_remote_code"]),
                            (SentenceTransformer.encode, ["prompt", "normalize_embeddings", "convert_to_numpy"]),
                            (cg, ["rtol", "atol", "maxiter"])]:
        parameters = inspect.signature(function).parameters
        if any(name not in parameters for name in names):
            raise RuntimeError(f"Unsupported API signature: {function.__qualname__}")
    fields = getattr(ModelConfig, "__dataclass_fields__", {})
    if "logprobs_mode" not in fields:
        raise RuntimeError("vLLM ModelConfig lacks processed-logprob support")
    SamplingParams(temperature=.8, top_p=1., top_k=0, min_p=0., max_tokens=8,
                   n=1, ignore_eos=False, seed=731000, extra_args={},
                   logprobs=5, logprob_token_ids=[0, 1])
    if not hasattr(AdapterLogitsProcessor, "new_req_logits_processor"):
        raise RuntimeError("Unsupported vLLM AdapterLogitsProcessor API")


def audit_environment(expected_vllm="0.20.0", package_root=None):
    root = Path(package_root or Path(__file__).parent)
    result = {"status": "FAIL", "python": sys.version, "platform": platform.platform(),
              "installation_performed": False, "errors": []}
    lines = (root.parent / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    try:
        result["dependencies"] = dependency_audit(requirements)
        result["errors"].extend(result["dependencies"]["errors"])
    except Exception as error:
        result["errors"].append(f"Dependency metadata: {type(error).__name__}: {error}")
    print("Checking actual imports (including lazy model and processor imports)...", flush=True)
    entries = collect_imports(root)
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        entries.append({"module": ".".join([root.name, *parts]), "symbol": None,
                        "locations": [str(path.relative_to(root))]})
    entries.extend([{"module": "transformers.models.qwen2.modeling_qwen2", "symbol": "Qwen2ForCausalLM", "locations": ["Qwen replay model"]},
                    {"module": "transformers.models.bert.modeling_bert", "symbol": "BertModel", "locations": ["BGE encoder model"]},
                    {"module": "torch.func", "symbol": "jvp", "locations": ["RMSNorm diagnostics"]},
                    {"module": "torch.func", "symbol": "vjp", "locations": ["RMSNorm diagnostics"]},
                    {"module": "torchvision", "symbol": None, "locations": ["vLLM native dependency / HF optional import"]},
                    {"module": "torchaudio", "symbol": None, "locations": ["vLLM native dependency"]}])
    result["imports"] = check_imports(entries)
    for row in result["imports"]:
        if row["status"] == "FAIL":
            result["errors"].append(f"Import {row['module']}:{row['symbol'] or '*'}: {row['error']}")
    try:
        api_audit()
        result["api_signatures"] = "PASS"
    except Exception as error:
        result["api_signatures"] = "NOT_PASSED"
        result["errors"].append(f"API: {type(error).__name__}: {error}")
    result["native"] = gpu_and_native_audit(expected_vllm)
    result["errors"].extend(result["native"]["errors"])
    result["errors"] = sorted(set(result["errors"]))
    result["status"] = "PASS" if not result["errors"] else "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON report destination")
    args = parser.parse_args()
    result = audit_environment()
    if args.output:
        from ._reused.imf_data_io import _atomic_text
        _atomic_text(Path(args.output), json.dumps(result, indent=2, allow_nan=False) + "\n")
    passed = sum(row["status"] == "PASS" for row in result["imports"])
    print(f"Import audit: {passed}/{len(result['imports'])} passed; environment {result['status']}")
    for error in result["errors"]:
        print(f"  - {error}")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
