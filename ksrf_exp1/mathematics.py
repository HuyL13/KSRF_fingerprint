"""Executable exact-enumeration checks of the experiment's score identity.

This tiny finite model is a mathematical diagnostic, never a generation backend.
It includes a sampled EOS and a length cap, with sequence-valued semantic features.
"""
from __future__ import annotations

import itertools
import numpy as np

from .common import output_dir, write_json


def check_score_identity():
    tau, eos, horizon = 0.8, 2, 3
    features = np.array([[0.2, -0.4], [-0.7, 0.5], [0.8, 0.9]], dtype=np.float64)
    semantic_table = {}
    for length in range(1, horizon + 1):
        for seq in itertools.product(range(3), repeat=length):
            # Fixed complete-sequence lookup, independent of eta and trajectory score.
            semantic_table[seq] = np.array([sum((i + 1) * (v - .3) for i, v in enumerate(seq)),
                                           (seq[0] == seq[-1]) + .1 * len(seq)**2])

    def softmax(logits):
        p = np.exp(logits - np.max(logits))
        return p / p.sum()

    def enumerate_model(eta):
        leaves, position_means = [], []

        def visit(prefix, probability, score, log_probability):
            if prefix and (prefix[-1] == eos or len(prefix) == horizon):
                leaves.append((prefix, probability, score, log_probability))
                return
            raw = np.array([.4 + .08 * sum(prefix), -.2 + .17 * len(prefix),
                            -.5 + .4 * len(prefix)])
            p = softmax(raw / tau + features @ eta)
            centered = features - p @ features
            position_means.append(p @ centered)
            for token in range(3):
                visit(prefix + (token,), probability * p[token],
                      score + centered[token], log_probability + np.log(p[token]))

        visit((), 1., np.zeros(2), 0.)
        return leaves, position_means

    leaves, position_means = enumerate_model(np.zeros(2))
    expectation = sum(prob * np.outer(semantic_table[seq], score)
                      for seq, prob, score, _ in leaves)
    eps = 1e-5
    finite_difference = np.zeros((2, 2))
    trajectory_error = 0.
    for coordinate in range(2):
        direction = np.eye(2)[coordinate] * eps
        plus, _ = enumerate_model(direction)
        minus, _ = enumerate_model(-direction)
        finite_difference[:, coordinate] = (
            sum(p * semantic_table[s] for s, p, _, _ in plus)
            - sum(p * semantic_table[s] for s, p, _, _ in minus)) / (2 * eps)
        for (_, _, score, _), pos, neg in zip(leaves, plus, minus):
            trajectory_error = max(trajectory_error,
                                   abs((pos[3] - neg[3]) / (2 * eps) - score[coordinate]))
    total_score = sum(prob * score for _, prob, score, _ in leaves)
    baseline_error = np.max(np.abs(np.outer(np.array([.2, -.8]), total_score)))
    # Catch temperature convention errors independently of the enumeration.
    raw = np.array([.3, -.9, .2])
    direction = np.array([.7, -.2])
    eta = .13
    temperature_error = np.max(np.abs(
        softmax((raw + tau * eta * (features @ direction)) / tau)
        - softmax(raw / tau + eta * (features @ direction))))
    result = {
        "status": "PASS", "trajectories": len(leaves),
        "max_derivative_error": float(np.max(np.abs(expectation - finite_difference))),
        "max_trajectory_logprob_derivative_error": float(trajectory_error),
        "position_centering_error": float(np.max(np.abs(position_means))),
        "baseline_expected_score_error": float(baseline_error),
        "temperature_scaling_error": float(temperature_error),
        "probability_sum": float(sum(p for _, p, _, _ in leaves)),
        "eos_covered": any(s[-1] == eos for s, *_ in leaves),
        "length_cap_covered": any(len(s) == horizon and s[-1] != eos for s, *_ in leaves),
    }
    errors = [value for key, value in result.items() if key.endswith("error")]
    if max(errors) > 1e-7 or abs(result["probability_sum"] - 1) > 1e-12:
        raise AssertionError(f"Exact score-function check failed: {result}")
    return result


def run(cfg):
    result = check_score_identity()
    result["causal_index_and_runtime_score"] = check_replay_indexing()
    write_json(output_dir(cfg) / "mathematical_checks.json", result)
    print(f"Score identity: PASS ({result['trajectories']} exact trajectories)", flush=True)


def check_replay_indexing():
    """Exercise the actual replay helpers on a fixed teacher-forced table."""
    import torch
    from types import SimpleNamespace
    from .replay_scores import response_logits, score_logits

    table = torch.tensor([[2., -1., .3], [.1, .2, -.3], [-.7, .8, .4],
                          [.5, -.2, .7], [-2., 3., 0.]], dtype=torch.float32)

    class FixedTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, input_ids, attention_mask, use_cache):
            if input_ids.tolist() != [[1, 0, 0, 1, 2]] or use_cache:
                raise AssertionError("Replay changed saved token IDs")
            if not torch.all(attention_mask == 1):
                raise AssertionError("Replay unexpectedly masked sampled tokens")
            return SimpleNamespace(logits=table[None, :, :])

    record = {"prompt_token_ids": [1, 0],
              "generated_token_ids_including_sampled_eos": [0, 1, 2]}
    logits = response_logits(FixedTeacher(), record)
    torch.testing.assert_close(logits, table[1:4], rtol=0, atol=0)
    features = torch.tensor([[.2, -.4], [-.7, .5], [.8, .9]])
    sampled = record["generated_token_ids_including_sampled_eos"]
    actual = score_logits(logits, sampled, features, .8, chunk_size=2).numpy()
    # Independent full-trajectory log-probability derivative in FP64,
    # explicitly including the third (sampled EOS) token.
    raw = table[1:4].double().numpy()
    feat = features.double().numpy()
    def log_probability(eta):
        shifted = raw / .8 + feat @ eta
        log_normalizer = np.log(np.exp(shifted).sum(axis=1))
        return float(sum(shifted[t, token] - log_normalizer[t] for t, token in enumerate(sampled)))
    eps = 1e-5
    expected = np.array([(log_probability(eps * axis) - log_probability(-eps * axis)) / (2 * eps)
                         for axis in np.eye(2)])
    error = float(np.max(np.abs(actual - expected)))
    if error > 2e-6:
        raise AssertionError("Runtime score helper disagrees with trajectory derivative")
    return {"status": "PASS", "max_derivative_error": error,
            "first_prediction_position": 1, "sampled_eos_included": True,
            "positions_tested": 3, "response_retokenized": False}
