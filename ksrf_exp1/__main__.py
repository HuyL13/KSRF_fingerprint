"""Process-isolated full runner. Stages are also individually inspectable."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from .common import file_checksum, load_config, output_dir, read_json, write_json


STAGES = {
    "preflight": ("preflight", "run"),
    "mathematics": ("mathematics", "run"),
    "prepare": ("data", "run"),
    "parity": ("validate_vllm_parity", "run"),
    "clean": ("clean_rollouts", "run"),
    "readout": ("readout_features", "run"),
    "replay": ("replay_scores", "run"),
    "semantic": ("semantic_features", "run"),
    "fit": ("fit_bridge", "run"),
    "gradient": ("validate_gradient", "run"),
    "delta": ("calibrate_delta", "run"),
    "perturb-smoke": ("validate_vllm_parity", "run_perturb_smoke"),
    "fd": ("validate_fd", "run"),
    "fd-generate": ("validate_fd", "run_generate"),
    "fd-analyze": ("validate_fd", "run_analyze"),
    "fd-kl": ("validate_fd", "run_kl"),
    "tangent": ("validate_tangent", "run"),
    "report": ("report", "run"),
}
PIPELINE = ["mathematics", "prepare", "parity", "clean", "readout", "replay",
            "semantic", "fit", "gradient", "delta", "perturb-smoke", "fd", "tangent", "report"]


def run_all(cfg):
    if (cfg["calibration_size"], cfg["heldout_size"], cfg["num_keys"], cfg["key_rank"]) != (5000, 1500, 16, 8):
        raise ValueError("Full Experiment 1 requires 5000 calibration, 1500 held-out, sixteen rank-eight keys")
    out = output_dir(cfg)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Full run requires a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "resolved_config.json"
    write_json(config_path, cfg)
    progress = {"status": "RUNNING", "completed_stages": [], "required_stages": PIPELINE}
    write_json(out / "pipeline_status.json", progress)
    for index, stage in enumerate(PIPELINE, 1):
        print(f"\n[{index}/{len(PIPELINE)}] {stage}", flush=True)
        try:
            subprocess.run([sys.executable, "-m", "ksrf_exp1", stage, "--config", str(config_path)], check=True)
            if stage == "prepare":
                frozen = read_json(out / "manifest.json")["config"]
                frozen["outputs"] = str(out)
                write_json(config_path, frozen)
        except subprocess.CalledProcessError as error:
            progress.update(status="FAILED", failed_stage=stage, exit_code=error.returncode)
            write_json(out / "pipeline_status.json", progress)
            raise
        progress["completed_stages"].append(stage)
        write_json(out / "pipeline_status.json", progress)
    report = read_json(out / "decision.json")
    progress.update(status="COMPLETE", decision=report["decision"])
    write_json(out / "pipeline_status.json", progress)
    # One final integrity index, consumed by the report reader/replay user.
    artifacts = {path.name: file_checksum(path) for path in sorted(out.iterdir())
                 if path.is_file() and path.name != "artifact_checksums.json"}
    write_json(out / "artifact_checksums.json", artifacts)
    print(f"\nAll {len(PIPELINE)} stages completed. Scientific decision: {report['decision']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="KSRF Experiment 1: full offline feasibility pipeline")
    parser.add_argument("stage", choices=["all", *STAGES])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output")
    parser.add_argument("--fd-count", type=int)
    args = parser.parse_args()
    cfg = (read_json(args.config) if Path(args.config).suffix == ".json" else load_config(args.config))
    if args.output:
        cfg["outputs"] = str(Path(args.output).resolve())
    if args.fd_count:
        cfg["fd_current_size"] = args.fd_count
    if args.stage == "all":
        run_all(cfg)
    else:
        module, function = STAGES[args.stage]
        getattr(importlib.import_module(f"ksrf_exp1.{module}"), function)(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
