# Follow-up whole-code audit

Reviewed the full project source after the Colab integration failures. No package
was installed, upgraded, downgraded or removed; no notebook environment change
was added to the experiment runner.

| Scope | Review and verification |
|---|---|
| `clean_rollouts`, `vllm_logit_processor`, `data`, reused prompt/replay helpers | Reviewed exact saved-token handling, sparse logprob parameters, EOS/cap handling and V1/V2 processor contracts against vLLM 0.32 source. |
| `validate_vllm_parity` | Found an unsupported exact cross-batch determinism assumption; corrected as described below. |
| `replay_scores` | Actual tiny BF16 Qwen2 replay matched direct causal logits; chunked Gamma matched the dense formula. |
| `semantic_features`, `readout_features` | Inspected frozen revisions, BGE encoding arguments, calibration-only PCA/whitening, actual lm_head extraction and precision. No additional concrete defect found. |
| `validate_tangent`, `calibrate_delta` | Actual frozen BF16 Qwen2 RMSNorm accepted FP32 JVP/VJP with its capture hook active; SciPy CG converged (relative normal-equation residual about 2.58e-7). Zero/nonzero token-KL checks passed. |
| `fit_bridge`, `key_and_statistic`, `validate_gradient` | Re-ran real NumPy/sklearn/SciPy synthetic fit, frozen baseline, sixteen-key bootstrap and controls. |
| `validate_fd`, `report`, `common` | Synthetic artifact flow through FD scores and report generation passed; frozen bridge/semantic transforms remained unchanged. Synthetic artifacts were temporary, not experimental evidence. |
| `__main__`, Bash, optional diagnostics | Checked stage wiring, required artifacts, compilation and Bash syntax. No environment audit or package installation runs in the Bash path. |

## Confirmed parity defect fixed

The old smoke test compared three unperturbed requests against their zero-eta
counterparts inside a nine-request batch, then demanded identical floating-point
logprobs and sampled trajectories. Initial parity also demanded exact trajectory
equality between singleton and batched inference. Default vLLM does not guarantee
these batch-invariant results; see the [official reproducibility guidance](https://docs.vllm.ai/en/v0.32.0/usage/reproducibility/).

The clean smoke reference now uses the same complete initial batch layout.
Comparisons require aligned prompt IDs, prompt token IDs and seeds, and compare
the specified diagnostic logprobs only while conditioning histories match. The
existing numerical tolerance is unchanged. Missing/nonfinite logprobs and
material same-context differences still fail. Full trajectory/float equality is
retained as diagnostic information, not an unsupported acceptance requirement.
Full HF replay comparisons, signed bias/temperature checks and EOS/cap tests
remain in the scientific gate.

Four regression checks passed: harmless numeric drift, trajectory divergence
with different later contexts, rejection of material same-context errors, and
rejection of mismatched request seeds. Independent re-review found no concrete
remaining issue in this fix.

## Limits

The local host has CPU-only runtime environments. These checks do not establish
that full Qwen 3B/BGE/vLLM GPU inference has completed on the user's Colab.
Full-checkpoint loading, actual GPU kernels and dataset-scale performance remain
to be observed on that runtime; no successful full-run result is claimed here.
