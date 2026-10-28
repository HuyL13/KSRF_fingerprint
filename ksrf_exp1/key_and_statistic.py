"""Haar keys and prompt-conditioned verifier coordinates."""
from __future__ import annotations

import numpy as np
from .common import require_finite


def make_keys(count=16, rank=8, seed=4417):
    # Separate RNG child streams, full O(r), no determinant +1 correction.
    keys = []
    for child in np.random.SeedSequence(seed).spawn(count):
        gaussian = np.random.default_rng(child).standard_normal((rank, rank))
        q, r = np.linalg.qr(gaussian)
        signs = np.where(np.diag(r) < 0, -1., 1.)
        keys.append(q * signs[None, :])
    return np.stack(keys)


def key_coordinates(z, key, bridge):
    b = require_finite("key coordinates", z @ key.T)
    w = b @ bridge["V8"].T @ bridge["L"].T
    a = b @ bridge["U8"].T
    return b, w, a


def baseline_prediction(prompt_embeddings, bridge):
    return prompt_embeddings @ bridge["baseline_coef"].T + bridge["baseline_intercept"]


def per_prompt_scores(features, bridge, key, residualized=True):
    b, w, _ = key_coordinates(features["z"], key, bridge)
    projection = features["phi"] @ bridge["U8"]
    if residualized:
        projection = projection - baseline_prediction(features["prompt_embedding"], bridge)
    residual = np.einsum("ij,ij->i", b, projection)
    derivative = np.einsum("ij,ij->i", features["Gamma"], w)
    return residual, derivative
