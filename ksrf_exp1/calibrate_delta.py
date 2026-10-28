"""Choose perturbation strength using calibration contexts only."""
from __future__ import annotations

import numpy as np
from .common import load_npz, output_dir, read_jsonl, write_json
from .key_and_statistic import key_coordinates


def token_kl(logits, bias, eta, temperature):
    import torch
    clean_logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    shifted_logp = torch.log_softmax(logits.float() / temperature + eta * bias, dim=-1)
    # KL(perturbed || clean), with exact full-vocabulary sums.
    return torch.sum(shifted_logp.exp() * (shifted_logp - clean_logp), dim=-1)


def run(cfg):
    import torch
    from scipy.optimize import brentq
    from .replay_scores import load_model, response_logits

    out = output_dir(cfg)
    features = load_npz(out / "calibration_features.npz")
    bridge = load_npz(out / "bridge.npz")
    primary = load_npz(out / "keys.npz")["R"][0]
    readout = load_npz(out / "readout_features.npz")
    lookup = {str(row["prompt_id"]): row for row in read_jsonl(out / "calibration_rollouts.jsonl")}
    count = min(cfg["delta_calibration_prompts"], len(features["prompt_id"]))
    _, weights, _ = key_coordinates(features["z"][:count], primary, bridge)
    fv = torch.as_tensor(readout["F_vocab"], device="cuda", dtype=torch.float32)
    model = load_model(cfg)
    # Cache only small-context logits on CPU to avoid repeated full-model replay
    # during scalar root-finding. This cache is ephemeral, never a disk artifact.
    contexts = []
    for index, pid in enumerate(features["prompt_id"][:count]):
        row = lookup[str(pid)]
        with torch.inference_mode():
            logits = response_logits(model, row)
            # Up to four naturally sampled positions per calibration response.
            positions = np.unique(np.linspace(0, len(logits)-1, min(4, len(logits))).astype(int))
            contexts.append((logits[positions].cpu(), (fv @ torch.as_tensor(weights[index],
                             device="cuda", dtype=torch.float32)).cpu()))
    del model, fv
    torch.cuda.empty_cache()

    def kl_values(eta):
        return torch.cat([token_kl(logits, bias, eta, cfg["temperature"])
                          for logits, bias in contexts]).numpy().astype(np.float64)

    target = cfg["delta_target_kl"]
    upper = 1.
    while kl_values(upper).mean() < target and upper < 1024:
        upper *= 2
    if kl_values(upper).mean() < target:
        raise ValueError("Could not bracket calibration KL target")
    delta = brentq(lambda eta: float(kl_values(eta).mean()) - target, 0., upper, xtol=1e-8)
    positive, negative = kl_values(delta), kl_values(-delta)
    write_json(out / "delta_calibration.json", {
        "delta": float(delta), "target_mean_token_kl": target,
        "positive_mean_token_kl": float(positive.mean()),
        "positive_p99_token_kl": float(np.quantile(positive, .99)),
        "negative_mean_token_kl": float(negative.mean()),
        "negative_p99_token_kl": float(np.quantile(negative, .99)),
        "calibration_prompt_ids": features["prompt_id"][:count].tolist(),
        "token_context_count": sum(len(logits) for logits, _ in contexts),
        "kl_direction": "KL(perturbed || clean)", "heldout_used": False})
    print(f"Calibration-only delta={delta:.6g}; mean token KL={positive.mean():.6g}", flush=True)
