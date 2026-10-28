"""Independent vLLM perturb-and-generate confirmation, including extension."""
from __future__ import annotations

import subprocess
import sys
import numpy as np

from .common import (load_npz, output_dir, read_json, read_jsonl, save_npz,
                     write_json, write_jsonl)
from .key_and_statistic import baseline_prediction, key_coordinates
from .validate_gradient import bootstrap_mean


def run_generate(cfg):
    from .clean_rollouts import generate
    out = output_dir(cfg)
    features = load_npz(out / "heldout_features.npz")
    bridge = load_npz(out / "bridge.npz")
    primary = load_npz(out / "keys.npz")["R"][0]
    count = min(cfg.get("fd_current_size", cfg["fd_initial_size"]), len(features["prompt_id"]))
    delta = read_json(out / "delta_calibration.json")["delta"]
    lookup = {str(row["prompt_id"]): row for row in read_jsonl(out / "heldout_rollouts.jsonl")}
    path = out / "fd_rollouts.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    done = {(str(row["source_prompt_id"]), row["eta_label"]) for row in existing}
    records, etas, weights, labels = [], [], [], []
    _, all_weights, _ = key_coordinates(features["z"][:count], primary, bridge)
    for branch, (label, eta) in enumerate([("minus", -delta), ("zero", 0.), ("plus", delta)]):
        for index, pid in enumerate(features["prompt_id"][:count]):
            if (str(pid), label) in done:
                continue
            source = dict(lookup[str(pid)])
            source["source_prompt_id"] = str(pid)
            source["prompt_id"] = f"{pid}:fd:{label}"
            # Disjoint from clean and other eta branches, stable across extension.
            source["sampling_seed"] = cfg["fd_sampling_seed"] + branch * 100000 + index
            records.append(source)
            etas.append(eta)
            weights.append(all_weights[index])
            labels.append(label)
    if records:
        generated = generate(cfg, records, etas=np.array(etas), weights=np.stack(weights), processor=True)
        if len(generated) != len(records):
            raise ValueError("Missing finite-difference vLLM requests")
        for source, result, label, eta in zip(records, generated, labels, etas):
            if str(result["prompt_id"]) != source["prompt_id"]:
                raise ValueError("Finite-difference request alignment failure")
            result.update(source_prompt_id=source["source_prompt_id"], eta_label=label, eta=float(eta))
        write_jsonl(path, existing + generated)
    print(f"Independent finite-difference rollouts available for {count} prompts per eta", flush=True)


def run_analyze(cfg):
    from .semantic_features import encode_texts, transform_embeddings
    out = output_dir(cfg)
    held = load_npz(out / "heldout_features.npz")
    bridge = load_npz(out / "bridge.npz")
    primary = load_npz(out / "keys.npz")["R"][0]
    count = min(cfg.get("fd_current_size", cfg["fd_initial_size"]), len(held["prompt_id"]))
    delta = read_json(out / "delta_calibration.json")["delta"]
    ids = list(map(str, held["prompt_id"][:count]))
    rows = read_jsonl(out / "fd_rollouts.jsonl")
    lookup = {(str(row["source_prompt_id"]), row["eta_label"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate finite-difference prompt/eta pairs")
    text_key = "response_text_without_special_tokens"
    exclusions = []
    valid = []
    for index, pid in enumerate(ids):
        missing = [label for label in ["minus", "zero", "plus"] if (pid, label) not in lookup]
        if missing:
            raise ValueError(f"Missing finite-difference branches for {pid}: {missing}")
        empty = [label for label in ["minus", "zero", "plus"] if not lookup[pid, label][text_key].strip()]
        if empty:
            exclusions.append({"prompt_id": pid, "reason": "empty response", "branches": empty})
        else:
            valid.append(index)
    if len(valid) < 2:
        raise ValueError("Insufficient nondegenerate finite-difference prompt triples")
    selected_ids = [ids[index] for index in valid]
    selected_prompts = held["prompt_embedding"][valid]
    selected_z = held["z"][valid]
    b, _, _ = key_coordinates(selected_z, primary, bridge)
    baseline = baseline_prediction(selected_prompts, bridge)
    metadata_path = out / "fd_embedding_metadata.json"
    previous_metadata = read_json(metadata_path) if metadata_path.exists() else {"truncations": []}
    scores, raw_scores, truncations, examples, lengths = {}, {}, previous_metadata["truncations"], {}, {}
    cache_path = out / "fd_embedding_cache.npz"
    if cache_path.exists():
        cache = load_npz(cache_path)
        embeddings = {key: value for key, value in zip(cache["ids"].tolist(), cache["embeddings"])}
    else:
        embeddings = {}
    pending = [lookup[pid, label] for label in ["minus", "zero", "plus"] for pid in selected_ids
               if lookup[pid, label]["prompt_id"] not in embeddings]
    if pending:
        encoded, truncated = encode_texts(cfg, [row[text_key] for row in pending])
        for row, embedding in zip(pending, encoded):
            embeddings[row["prompt_id"]] = embedding
        truncations.extend(dict(item, prompt_id=pending[item["index"]]["prompt_id"])
                           for item in truncated)
        save_npz(cache_path, ids=np.array(list(embeddings)), embeddings=np.stack(list(embeddings.values())))
    write_json(metadata_path, {"truncations": truncations, "cached_responses": len(embeddings)})
    for label in ["minus", "zero", "plus"]:
        branch = [lookup[pid, label] for pid in selected_ids]
        response = np.stack([embeddings[row["prompt_id"]] for row in branch])
        phi, _ = transform_embeddings(cfg, selected_prompts, response)
        raw_scores[label] = np.einsum("ij,ij->i", b, phi @ bridge["U8"])
        scores[label] = np.einsum("ij,ij->i", b, phi @ bridge["U8"] - baseline)
        examples[label] = [{"prompt_id": row["source_prompt_id"],
                            "prompt": row["prompt_text"], "response": row[text_key],
                            "stop_reason": row["stop_reason"]} for row in branch[:5]]
        token_counts = [len(row["generated_token_ids_including_sampled_eos"]) for row in branch]
        lengths[label] = {"mean": float(np.mean(token_counts)), "p95": float(np.quantile(token_counts, .95)),
                          "sampled_eos": sum(row["stop_reason"] == "sampled_eos" for row in branch),
                          "length_cap": sum(row["stop_reason"] == "length_cap" for row in branch)}
    fd_contributions = (scores["plus"] - scores["minus"]) / (2 * delta)
    fd = bootstrap_mean(fd_contributions, cfg, cfg["selection_seed"] + 19)
    gradient = read_json(out / "gradient_bootstrap.json")["keys"][0]["residualized"]
    compatible = max(fd["ci_low"], gradient["ci_low"]) <= min(fd["ci_high"], gradient["ci_high"])
    result = {"prompt_count": count, "valid_prompt_triples": len(valid), "delta": delta,
              "D_fd": fd, "D_score": gradient, "compatible_intervals": bool(compatible),
              "scores": {label: bootstrap_mean(array, cfg) for label, array in scores.items()},
              "non_residualized_scores": {label: bootstrap_mean(array, cfg) for label, array in raw_scores.items()},
              "quality_examples": examples, "lengths": lengths, "exclusions": exclusions,
              "encoder_truncations": truncations,
              "generation_backend": "vllm", "independent_sampling": True,
              "extension_rule": "500 first; extend to 1500 if positive trend/compatible uncertainty is not established",
              "positive_trend": bool(fd["ci_low"] > 0),
              "output_quality_requires_human_inspection": True}
    write_json(out / "finite_difference.json", result)
    print(f"Finite difference: {count} prompts, D_fd={fd['mean']:.6g}, 95% CI [{fd['ci_low']:.6g}, {fd['ci_high']:.6g}]", flush=True)


def run_kl(cfg):
    import torch
    from .calibrate_delta import token_kl
    from .replay_scores import load_model, response_logits
    out = output_dir(cfg)
    held = load_npz(out / "heldout_features.npz")
    bridge = load_npz(out / "bridge.npz")
    primary = load_npz(out / "keys.npz")["R"][0]
    _, weights, _ = key_coordinates(held["z"], primary, bridge)
    by_id = {str(pid): w for pid, w in zip(held["prompt_id"], weights)}
    fv = torch.as_tensor(load_npz(out / "readout_features.npz")["F_vocab"], device="cuda", dtype=torch.float32)
    result = read_json(out / "finite_difference.json")
    rollouts = read_jsonl(out / "fd_rollouts.jsonl")
    selected = set(map(str, held["prompt_id"][:cfg["delta_calibration_prompts"]]))
    model = load_model(cfg)
    diagnostics = {}
    for label in ["minus", "zero", "plus"]:
        values = []
        for row in rollouts:
            if row["eta_label"] != label or str(row["source_prompt_id"]) not in selected:
                continue
            with torch.inference_mode():
                logits = response_logits(model, row)
                positions = np.unique(np.linspace(0, len(logits)-1, min(4, len(logits))).astype(int))
                bias = fv @ torch.as_tensor(by_id[str(row["source_prompt_id"])], device="cuda", dtype=torch.float32)
                values.extend(token_kl(logits[positions], bias, row["eta"], cfg["temperature"]).cpu().tolist())
        if not values:
            raise ValueError(f"Missing independent token KL contexts for {label}")
        diagnostics[label] = {"mean": float(np.mean(values)), "p99": float(np.quantile(values, .99)),
                              "contexts": len(values), "direction": "KL(perturbed || clean)"}
    result["token_kl"] = diagnostics
    result["token_kl_scope"] = "Up to 4 positions in each of the first 32 fixed held-out prompts per eta, replayed on independent generated contexts"
    write_json(out / "finite_difference.json", result)


def run(cfg):
    # A separate interpreter owns each GPU runtime and its workers.
    config_path = output_dir(cfg) / "resolved_config.json"
    sizes = [min(cfg["fd_initial_size"], cfg["heldout_size"])]
    if sizes[0] < cfg["heldout_size"]:
        sizes.append(cfg["heldout_size"])
    for size in sizes:
        for stage in ["fd-generate", "fd-analyze"]:
            subprocess.run([sys.executable, "-m", "ksrf_exp1", stage, "--config", str(config_path),
                            "--fd-count", str(size)], check=True)
        result = read_json(output_dir(cfg) / "finite_difference.json")
        if result["positive_trend"] and result["compatible_intervals"]:
            break
    subprocess.run([sys.executable, "-m", "ksrf_exp1", "fd-kl", "--config", str(config_path)], check=True)
