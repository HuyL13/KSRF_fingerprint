"""Held-out, key-conditioned derivatives with prompt-level bootstrap."""
from __future__ import annotations

import numpy as np
from .common import load_npz, output_dir, read_json, write_json
from .key_and_statistic import baseline_prediction, key_coordinates, per_prompt_scores


def bootstrap_mean(values, cfg, seed=20261028):
    from scipy.stats import bootstrap
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite values for at least two prompts")
    result = bootstrap((values,), np.mean, axis=0, vectorized=True,
                       n_resamples=cfg["bootstrap_replicates"], batch=32,
                       confidence_level=.95, method="percentile", random_state=seed)
    return {"mean": np.mean(values, axis=0).tolist(),
            "ci_low": np.asarray(result.confidence_interval.low).tolist(),
            "ci_high": np.asarray(result.confidence_interval.high).tolist()}


def run(cfg):
    out = output_dir(cfg)
    held = load_npz(out / "heldout_features.npz")
    cal = load_npz(out / "calibration_features.npz")
    if set(held["prompt_id"].tolist()) & set(cal["prompt_id"].tolist()):
        raise ValueError("Calibration/held-out prompt leakage")
    bridge = load_npz(out / "bridge.npz")
    random_bridge = load_npz(out / "random_bridge.npz")
    keys = load_npz(out / "keys.npz")["R"]
    rng = np.random.default_rng(cfg["selection_seed"] + 11)
    permutation = rng.permutation(len(held["phi"]))
    derivatives, raw_derivatives, permutation_derivatives, random_derivatives = [], [], [], []
    for key in keys:
        residual, direction = per_prompt_scores(held, bridge, key)
        raw, _ = per_prompt_scores(held, bridge, key, residualized=False)
        random_residual, random_direction = per_prompt_scores(held, random_bridge, key)
        _, native_w, _ = key_coordinates(held["z"], key, bridge)
        _, random_w, _ = key_coordinates(held["z"], key, random_bridge)
        learned_norm2 = np.einsum("ij,jk,ik->i", native_w, bridge["F"], native_w)
        random_norm2 = np.einsum("ij,jk,ik->i", random_w, bridge["F"], random_w)
        if (random_norm2 <= 0).any() or (learned_norm2 <= 0).any():
            raise ValueError("Degenerate native-Fisher norm in random control")
        random_direction = random_direction * np.sqrt(learned_norm2 / random_norm2)
        b_perm, w_perm, _ = key_coordinates(held["z"][permutation], key, bridge)
        perm_residual = np.einsum("ij,ij->i", b_perm,
                                  held["phi"] @ bridge["U8"] -
                                  baseline_prediction(held["prompt_embedding"][permutation], bridge))
        perm_direction = np.einsum("ij,ij->i", held["Gamma"], w_perm)
        derivatives.append(residual * direction)
        raw_derivatives.append(raw * direction)
        permutation_derivatives.append(perm_residual * perm_direction)
        random_derivatives.append(random_residual * random_direction)
    values = np.stack(derivatives, axis=1)
    controls = {"residualized": values,
                "non_residualized": np.stack(raw_derivatives, axis=1),
                "permuted_prompt_pairing": np.stack(permutation_derivatives, axis=1),
                "random_subspaces": np.stack(random_derivatives, axis=1)}
    summary = {name: bootstrap_mean(array, cfg) for name, array in controls.items()}
    rows = []
    for index in range(len(keys)):
        row = {"key_index": index, "primary": index == 0}
        for name, estimate in summary.items():
            row[name] = {field: estimate[field][index] for field in estimate}
        rows.append(row)
    rank = cfg["key_rank"]
    held_c = (held["phi"] - bridge["mu_phi"]).T @ held["Gamma"] / len(held["phi"])
    held_alignment = float(np.trace(bridge["U8"].T @ held_c @ bridge["L"] @ bridge["V8"]) / rank)
    # This is a held-out evaluation of the frozen operator, never a refit.
    projected = bridge["U8"].T @ held_c @ bridge["L"] @ bridge["V8"]
    result = {"keys": rows, "heldout_count": len(values),
              "bootstrap_replicates": cfg["bootstrap_replicates"],
              "bootstrap_unit": "complete prompt-response pair", "primary_key_index": 0,
              "key_average": bootstrap_mean(values.mean(axis=1), cfg),
              "learned_minus_random": bootstrap_mean((values - controls["random_subspaces"]).mean(axis=1), cfg),
              "learned_minus_permuted": bootstrap_mean((values - controls["permuted_prompt_pairing"]).mean(axis=1), cfg),
              "heldout_projected_C_trace_per_rank": held_alignment,
              "heldout_projected_C_singular_values": np.linalg.svd(projected, compute_uv=False).tolist(),
              "normalized_heldout_alignment": held_alignment / max(float(bridge["singular_values"][:rank].mean()), 1e-30),
              "note": "Population key-average identity is not a fixed-key guarantee. No key selection."}
    write_json(out / "gradient_bootstrap.json", result)
    print(f"Held-out derivatives and controls evaluated for {len(keys)} predeclared keys", flush=True)
