"""Calibration-only response/readout operator fitting and stability controls."""
from __future__ import annotations

import numpy as np
from .common import load_npz, output_dir, require_finite, save_npz, write_json
from .key_and_statistic import make_keys


def fit_operator(phi, gamma, rank=8, shrinkage=.05):
    phi = require_finite("phi", phi).astype(np.float64)
    gamma = require_finite("Gamma", gamma).astype(np.float64)
    if len(phi) != len(gamma) or len(phi) < 2 or min(phi.shape[1], gamma.shape[1]) < rank:
        raise ValueError("Insufficient/misaligned calibration dimensions")
    mu = phi.mean(axis=0)
    c = (phi - mu).T @ gamma / len(phi)
    fisher = gamma.T @ gamma / len(gamma)
    fisher = (fisher + fisher.T) / 2
    ridge = shrinkage * np.trace(fisher) / fisher.shape[0]
    if ridge <= 0:
        raise ValueError("Degenerate zero trajectory Fisher")
    f_reg = fisher + ridge * np.eye(fisher.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(f_reg)
    if eigenvalues.min() <= 0:
        raise ValueError("Regularized Fisher is not positive definite")
    inv_sqrt = (eigenvectors * eigenvalues**-.5) @ eigenvectors.T
    u, singular_values, vh = np.linalg.svd(c @ inv_sqrt, full_matrices=False)
    result = dict(C=c, F=fisher, F_reg=f_reg, L=inv_sqrt, U8=u[:, :rank],
                  V8=vh.T[:, :rank], singular_values=singular_values,
                  fisher_eigenvalues=eigenvalues, mu_phi=mu,
                  condition_number=np.array(eigenvalues[-1] / eigenvalues[0]))
    for name, value in result.items():
        require_finite(name, value)
    for basis in [result["U8"], result["V8"]]:
        np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=1e-8)
    return result


def run(cfg):
    from sklearn.linear_model import Ridge
    from scipy.linalg import subspace_angles

    out = output_dir(cfg)
    cal = load_npz(out / "calibration_features.npz")
    rank, shrink = cfg["key_rank"], cfg["fisher_shrinkage"]
    bridge = fit_operator(cal["phi"], cal["Gamma"], rank, shrink)
    baseline = Ridge(alpha=cfg["ridge_alpha"], fit_intercept=True, solver="svd")
    baseline.fit(cal["prompt_embedding"].astype(np.float64), cal["phi"] @ bridge["U8"])
    bridge.update(baseline_coef=baseline.coef_, baseline_intercept=baseline.intercept_)
    save_npz(out / "bridge.npz", **bridge)
    save_npz(out / "keys.npz", R=make_keys(cfg["num_keys"], rank, cfg["key_seed"]))
    middle = len(cal["phi"]) // 2
    half_a = fit_operator(cal["phi"][:middle], cal["Gamma"][:middle], rank, shrink)
    half_b = fit_operator(cal["phi"][middle:], cal["Gamma"][middle:], rank, shrink)
    rng = np.random.default_rng(cfg["projection_seed"] + 1)
    perm = rng.permutation(len(cal["phi"]))
    shuffled = fit_operator(cal["phi"][perm], cal["Gamma"], rank, shrink)
    # Random semantic/native-Fisher-normalized readout control is saved once,
    # before the held-out statistic is evaluated.
    random_u, _ = np.linalg.qr(rng.standard_normal(bridge["U8"].shape))
    random_v, _ = np.linalg.qr(rng.standard_normal(bridge["V8"].shape))
    control = dict(bridge, U8=random_u, V8=random_v)
    control_baseline = Ridge(alpha=cfg["ridge_alpha"], fit_intercept=True, solver="svd")
    control_baseline.fit(cal["prompt_embedding"].astype(np.float64), cal["phi"] @ random_u)
    control.update(baseline_coef=control_baseline.coef_,
                   baseline_intercept=control_baseline.intercept_)
    save_npz(out / "random_bridge.npz", **control)
    save_npz(out / "bridge_halves.npz", **{f"a_{k}": v for k, v in half_a.items()},
             **{f"b_{k}": v for k, v in half_b.items()}, shuffled_C=shuffled["C"],
             shuffled_singular_values=shuffled["singular_values"])
    semantic_angles = np.degrees(subspace_angles(half_a["U8"], half_b["U8"]))
    native_angles = np.degrees(subspace_angles(half_a["L"] @ half_a["V8"],
                                             half_b["L"] @ half_b["V8"]))
    diagnostics = {
        "semantic_principal_angles_degrees": semantic_angles.tolist(),
        "native_principal_angles_degrees": native_angles.tolist(),
        "singular_values": bridge["singular_values"].tolist(),
        "half_a_singular_values": half_a["singular_values"].tolist(),
        "half_b_singular_values": half_b["singular_values"].tolist(),
        "shuffled_singular_values": shuffled["singular_values"].tolist(),
        "leading_trace": float(bridge["singular_values"][:rank].sum()),
        "shuffled_leading_trace": float(shuffled["singular_values"][:rank].sum()),
        "key_average_population_identity": float(bridge["singular_values"][:rank].mean()),
        "fisher_eigenvalues": bridge["fisher_eigenvalues"].tolist(),
        "condition_number": float(bridge["condition_number"]),
        "calibration_count": len(cal["phi"]), "primary_key_index": 0,
        "primary_key_seed": cfg["key_seed"], "ridge_alpha": cfg["ridge_alpha"],
        "random_control_norm": "Per-prompt direction rescaled to match learned w^T F w during held-out evaluation",
        "stability_angle_limit_degrees": cfg["stability_max_angle_degrees"],
    }
    write_json(out / "key_diagnostics.json", diagnostics)
    print("Bridge, frozen baseline, 16 keys and stability controls fitted", flush=True)
