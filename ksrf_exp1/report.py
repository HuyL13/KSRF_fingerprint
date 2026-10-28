"""Evidence-based feasibility decision and a self-contained research report."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .common import load_npz, output_dir, read_json, read_jsonl, write_json


def interval(estimate):
    return f"{estimate['mean']:.6g} [{estimate['ci_low']:.6g}, {estimate['ci_high']:.6g}]"


def decide(cfg, mathematical, parity, smoke, stability, gradient, fd, tangent, quality_review=None):
    blockers, uncertainty = [], []
    for name, evidence in [("score-function mathematics", mathematical),
                           ("vLLM/HF distribution parity", parity),
                           ("real-readout perturbation smoke", smoke)]:
        if evidence.get("status") != "PASS":
            blockers.append(name + " failed or is missing")
    primary = gradient["keys"][0]["residualized"]
    if primary["ci_high"] < 0:
        blockers.append("Predeclared primary key has a negative held-out derivative with 95% CI below zero")
    elif primary["ci_low"] <= 0:
        uncertainty.append("Primary-key derivative confidence interval includes zero")
    if gradient["key_average"]["ci_low"] <= 0:
        uncertainty.append("Key-averaged positive derivative is not established")
    if sum(row["residualized"]["mean"] > 0 for row in gradient["keys"]) <= len(gradient["keys"]) / 2:
        uncertainty.append("Most diagnostic keys do not have positive point estimates")
    if stability["leading_trace"] <= stability["shuffled_leading_trace"]:
        uncertainty.append("Learned operator does not exceed shuffled singular strength")
    for name in ["learned_minus_random", "learned_minus_permuted"]:
        if gradient[name]["ci_low"] <= 0:
            uncertainty.append(f"Held-out superiority to {name.replace('learned_minus_', '')} control is uncertain")
    for name in ["semantic_principal_angles_degrees", "native_principal_angles_degrees"]:
        if max(stability[name]) > cfg["stability_max_angle_degrees"]:
            uncertainty.append(f"Calibration half-split {name} exceed the predeclared stability limit")
    if fd["D_fd"]["ci_high"] < 0 and primary["ci_low"] > 0:
        blockers.append("Independent decoding trend is significantly negative despite positive predicted derivative")
    elif fd["D_fd"]["ci_low"] <= 0:
        uncertainty.append("Independent decoding is underpowered or has no positive trend")
    if not fd["compatible_intervals"]:
        uncertainty.append("Finite-difference and score-function confidence intervals do not overlap")
    residual = tangent.get("relative_tangent_residual", {})
    residual_p95 = residual.get("p95", tangent.get("residual_p95"))
    if residual_p95 is None:
        raise ValueError("Tangent diagnostic is missing p95 residual")
    residual_median = residual.get("median", tangent.get("residual_median"))
    if residual_median is None:
        raise ValueError("Tangent diagnostic is missing median residual")
    if residual_median > cfg["tangent_residual_warning"]:
        blockers.append("Native-readout tangent residual exceeds the predeclared warning level")
    elif residual_p95 > cfg["tangent_residual_warning"]:
        uncertainty.append("Upper-tail native-readout tangent residual is large")
    details = tangent.get("details", [])
    if not details or any(row.get("solver_info", -1) != 0 or
                          not np.isfinite(row.get("normal_equation_relative_residual", float("nan"))) or
                          row.get("normal_equation_relative_residual", 1.) > 1e-3
                          for row in details):
        uncertainty.append("One or more damped tangent solves did not converge")
    if fd["exclusions"]:
        uncertainty.append("Finite-difference generation contains excluded empty responses")
    # Text fluency/meaning is not certified by length statistics or an invented
    # automatic metric. The pipeline still completes with an honest decision.
    quality_review = quality_review or {}
    reviewed = quality_review.get("reviewed") is True and bool(quality_review.get("reviewer"))
    if reviewed and quality_review.get("no_obvious_degeneration") is False:
        blockers.append("Researcher inspection found obvious output degeneration")
    elif not blockers and not uncertainty and not (
            reviewed and quality_review.get("no_obvious_degeneration") is True):
        uncertainty.append("All numerical gates pass; qualitative output examples still require research review before GO")
    decision = "NO-GO" if blockers else "INCONCLUSIVE" if uncertainty else "GO"
    return {"decision": decision, "reasons": blockers + uncertainty,
            "scope": "Experiment 1 feasibility only; no implantation or ownership claim",
            "qualitative_review_required": not reviewed,
            "gate_policy": "Conservative: statistical uncertainty is not proof of failure"}


def run(cfg):
    out = output_dir(cfg)
    names = {"manifest": "manifest.json", "environment": "environment.json",
             "math": "mathematical_checks.json", "parity": "vllm_parity.json",
             "smoke": "perturb_smoke.json", "stability": "key_diagnostics.json",
             "gradient": "gradient_bootstrap.json", "delta": "delta_calibration.json",
             "fd": "finite_difference.json", "tangent": "tangent_diagnostics.json"}
    evidence = {name: read_json(out / path) for name, path in names.items()}
    # Reports are never written from partial runs or missing analytical artifacts.
    for filename in ["bridge.npz", "bridge_halves.npz", "random_bridge.npz", "keys.npz",
                     "semantic_transform.npz", "semantic_metadata.json", "readout_features.npz",
                     "calibration_features.npz", "heldout_features.npz"]:
        if not (out / filename).is_file():
            raise ValueError(f"Missing required report artifact: {filename}")
    features = {name: load_npz(out / f"{name}_features.npz") for name in ["calibration", "heldout"]}
    bridge = load_npz(out / "bridge.npz")
    for name, data in features.items():
        for field in ["Gamma", "phi", "z", "prompt_embedding", "response_embedding", "prompt_id"]:
            if field not in data:
                raise ValueError(f"Incomplete {name} features: missing {field}")
    quality_path = out / "quality_review.json"
    quality_review = read_json(quality_path) if quality_path.exists() else None
    decision = decide(cfg, evidence["math"], evidence["parity"], evidence["smoke"],
                      evidence["stability"], evidence["gradient"], evidence["fd"], evidence["tangent"], quality_review)
    write_json(out / "decision.json", decision)
    frozen = evidence["manifest"]["config"]
    lines = ["# KSRF Experiment 1", "", f"**Decision: {decision['decision']}**", ""]
    lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.extend(["", "## Data and runtime", "", "| Item | Value |", "|---|---|"])
    for key in ["model", "model_revision", "tokenizer_revision", "encoder", "encoder_revision", "dataset", "dataset_revision", "vllm_version"]:
        lines.append(f"| {key} | `{frozen[key]}` |")
    lines.extend([f"| Processor API | `{evidence['environment']['processor_api']}` |",
                  f"| GPU | {', '.join(evidence['environment']['gpu_names'])} |",
                  f"| Generation batch size | {cfg['generation_batch_size']} |",
                  f"| GPU memory utilization | {cfg['gpu_memory_utilization']} |",
                  f"| Tensor parallel size | {cfg['tensor_parallel_size']} |", "",
                  "Exact runner/sampling settings and dependency versions are in `manifest.json`,",
                  "`environment.json` and each rollout record. Temperature 0.8, top_p 1, top_k 0,",
                  "min_p 0, max_tokens 256, n 1; no generation-config overrides or clean custom processor.", "",
                  "| Split | Selected | Analyzed | EOS | Length cap | Mean tokens | p95 tokens |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for split in ["calibration", "heldout"]:
        rows = read_jsonl(out / f"{split}_rollouts.jsonl")
        lengths = [len(row["generated_token_ids_including_sampled_eos"]) for row in rows]
        lines.append(f"| {split} | {len(rows)} | {len(features[split]['phi'])} | "
                     f"{sum(row['stop_reason']=='sampled_eos' for row in rows)} | "
                     f"{sum(row['stop_reason']=='length_cap' for row in rows)} | "
                     f"{np.mean(lengths):.2f} | {np.quantile(lengths,.95):.2f} |")
    lines.extend(["", "Skipped prompts are in `skipped_prompts.jsonl`; replay exclusion reasons and",
                  "encoder truncations are recorded in the extraction metadata files.", "",
                  "## Mathematical and generation checks", "",
                  f"- Exact enumerated score identity: **{evidence['math']['status']}**, maximum derivative error "
                  f"{evidence['math']['max_derivative_error']:.4g}.",
                  f"- vLLM/HF parity, EOS, temperature, batch and request checks: **{evidence['parity']['status']}**.",
                  f"- Fitted-readout signed perturbation/no-op smoke: **{evidence['smoke']['status']}**.",
                  "- HF replay is a measured numerical approximation across different GPU kernels;",
                  "  it is not asserted bit-exact to vLLM. The configured error tolerance is recorded.",
                  "- Detailed evidence: `mathematical_checks.json`, `vllm_parity.json`, `perturb_smoke.json`.", "",
                  "## Operator and stability", "",
                  f"C shape: `{bridge['C'].shape}`; F shape: `{bridge['F'].shape}`. Full matrices, regularized Fisher,",
                  "eigenvalues, whitening operator and singular bases are in `bridge.npz`.", "",
                  f"Fisher condition number: {float(bridge['condition_number']):.6g}.",
                  f"Singular spectrum: {evidence['stability']['singular_values']}.",
                  f"Semantic half-split angles (degrees): {evidence['stability']['semantic_principal_angles_degrees']}.",
                  f"Native readout half-split angles (degrees): {evidence['stability']['native_principal_angles_degrees']}.",
                  f"True/shuffled leading singular traces: {evidence['stability']['leading_trace']:.6g} / "
                  f"{evidence['stability']['shuffled_leading_trace']:.6g}.",
                  f"Normalized held-out projected C alignment: {evidence['gradient']['normalized_heldout_alignment']:.6g}.",
                  "Half-fit matrices and shuffled spectrum are in `bridge_halves.npz`.", "",
                  "## Held-out derivatives: all 16 predeclared keys", "",
                  "Each cell is mean [95% prompt-bootstrap CI]. Key 0 is primary; no post hoc key selection.", "",
                  "| Key | Residualized | No baseline | Permuted prompts | Random subspaces |",
                  "|---|---|---|---|---|"])
    for row in evidence["gradient"]["keys"]:
        label = str(row["key_index"]) + (" (primary)" if row["primary"] else "")
        lines.append("| " + " | ".join([label] + [interval(row[key]) for key in
                     ["residualized", "non_residualized", "permuted_prompt_pairing", "random_subspaces"]]) + " |")
    lines.extend(["", f"Key average: {interval(evidence['gradient']['key_average'])}.",
                  f"Learned minus random: {interval(evidence['gradient']['learned_minus_random'])}.",
                  f"Learned minus permuted: {interval(evidence['gradient']['learned_minus_permuted'])}.",
                  "The population key-averaged identity does not guarantee a positive fixed-key or per-prompt derivative.", "",
                  "## Independent finite differences", "",
                  f"Calibration-only delta: {evidence['delta']['delta']:.6g}; target token KL 0.002 nats.",
                  f"Prompts per eta: {evidence['fd']['prompt_count']}; valid triples: {evidence['fd']['valid_prompt_triples']}.",
                  f"D_fd: {interval(evidence['fd']['D_fd'])}; independent D_score: {interval(evidence['fd']['D_score'])}.",
                  f"Compatible intervals: {evidence['fd']['compatible_intervals']}.", "",
                  "| Eta | Score [95% CI] | Mean KL | p99 KL | Mean tokens | p95 tokens |",
                  "|---|---|---:|---:|---:|---:|"])
    for label in ["minus", "zero", "plus"]:
        kl = evidence["fd"]["token_kl"][label]
        lengths = evidence["fd"]["lengths"][label]
        lines.append(f"| {label} | {interval(evidence['fd']['scores'][label])} | {kl['mean']:.6g} | "
                     f"{kl['p99']:.6g} | {lengths['mean']:.2f} | {lengths['p95']:.2f} |")
    lines.extend(["", evidence["fd"]["token_kl_scope"] + ".", "",
                  "Token/stop diagnostics and unresidualized scores are in `finite_difference.json`.",
                  "Inspect the following matched examples for semantic degeneration; automatic length statistics do not certify fluency.", ""])
    for index in range(min(3, len(evidence["fd"]["quality_examples"]["zero"]))):
        lines.extend([f"### Example {index+1}", "", evidence["fd"]["quality_examples"]["zero"][index]["prompt"], ""])
        for label in ["minus", "zero", "plus"]:
            response = evidence["fd"]["quality_examples"][label][index]["response"]
            lines.extend([f"**{label}:**", "", "> " + response.replace("\n", "\n> "), ""])
    lines.extend(["## Native RMSNorm tangent realizability", "",
                  "Actual final RMSNorm JVP/VJP with a damped least-squares inverse; no dense hidden Jacobian.",
                  "Full diagnostics (including median/p95 residual and required pre-norm shift norm):", "", "```json",
                  __import__("json").dumps(evidence["tangent"], indent=2), "```", "",
                  "## Scope and interpretation", "",
                  "No LoRA, SFT/DPO, implantation, online generation API, paraphrase test, ownership TPR/FPR,",
                  "quantization-robustness or removal-resistance claim was made. No verifier threshold was tuned.",
                  "All preprocessing, keys, baseline and delta were frozen using calibration only.",
                  "Statistical uncertainty warrants additional independent data; it is not alone a redesign verdict.", ""])
    (out / "EXP1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Research report written: {decision['decision']}", flush=True)
