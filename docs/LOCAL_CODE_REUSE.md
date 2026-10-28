# Actual reuse of local source code

The initial implementation reused library APIs but did **not** incorporate local
project code. That did not satisfy the user's intended meaning of code reuse.
This follow-up audits local candidates and replaces duplicated implementation
with explicit source excerpts used by the live pipeline.

## Scope and limits of the audit

A filename/content scan covered **482 project Python files in 13 local
repositories**, including their project tests, while excluding environments,
large third-party trees and repeated vendored copies from that count. This is a
search/inventory count, **not a claim that every file was read line by line**.
Candidate functions were then inspected directly. The RIPPLE archive and selected
vendor sources were inspected separately and are not part of the 482-file count.

| Local repository | Project Python files inventoried |
|---|---:|
| LLM-Fingerprint-and-Attacks | 12 |
| MEraser | 5 |
| phaseA_fingerprint | 12 |
| if_awq_tier0 | 30 |
| imf_ptq | 49 |
| imf-tier0-ptq | 72 |
| phase1_if_analysis | 40 |
| if-sft-turboquant-lab | 35 |
| rtn2to4 | 37 |
| RBVT-squeeze | 50 |
| tailor-q-guidedquant | 66 |
| quantization_attack | 55 |
| unlearning-quantization-attack | 19 |

Searches covered embeddings/BGE, PCA/whitening, Jacobians/Fisher, teacher forcing,
vLLM/logits processors, bootstrap/subspaces, atomic I/O and provenance. The initial
vLLM upstream source/API review is documented separately in `UPSTREAM.md`.

## Code actually incorporated and called

These are actual extracted function bodies, not newly written equivalents or
imports of an installed numerical library. They are self-contained under
`ksrf_exp1/_reused`, so Colab does not need sibling local repositories.

| Original source | Functions | Destination | Live caller |
|---|---|---|---|
| `phase1_if_analysis/src/phase1/runtime.py` | `device_of`, `prompt_ids`, `teacher_forced_logits` | `_reused/phase1_runtime.py` | `data.py`, `replay_scores.py` |
| `imf-tier0-ptq/src/imf_tier0/data/io.py` | `_atomic_text`, `read_jsonl` | `_reused/imf_data_io.py` | `common.py`, hence pipeline artifact readers/writers |
| `imf-tier0-ptq/src/imf_tier0/manifest.py` | `sha256_file` | `_reused/imf_provenance.py` | `common.file_checksum`, final artifact integrity index |

**Four function bodies are unchanged.** `prompt_ids` now explicitly requests
`return_dict=False` from chat templating and validates integer token IDs, because
Transformers 5 otherwise returns mapping keys when the original function calls
`list(...)`. `teacher_forced_logits` retains the
original token construction, context check, attention mask, inference call and
causal slicing. Two adaptations are explicitly marked:

1. Keep the FP32 logits on the model device instead of converting the entire
   vocabulary tensor to CPU NumPy and copying it back for Gamma accumulation.
2. Allow a model fixture without `.config` while retaining the context bound for
   real HF models.

The KSRF replay wrapper supplies saved `prompt_token_ids` and `target_token_ids`;
the reused function's text-tokenization fallbacks are never used during replay.
Sampled EOS is preserved. Only the three relevant functions were extracted from
the phase1 runtime; its HF generation, quantization and verifier code are excluded.

The JSON wrappers keep KSRF's strict `allow_nan=False` policy and delegate durable
writing to the unchanged local `_atomic_text`. The reused reader additionally
rejects non-object JSONL rows. A failed serialization preserves existing artifacts
and leaves no partially written temporary file.

Source revisions:

- https://github.com/HuyL13/phase1_if_analysis/tree/a7ca5abe5d1266415af4d0d34fa4928f14695efd
- https://github.com/HuyL13/imf-tier0-ptq/tree/94931f65e1eb3454200174118db10f06363a3ff5

`local_reuse_sources.json` records source file hashes, original function hashes,
line numbers and destination paths. Function-body comparisons verify the five
unchanged excerpts rather than relying on a prose claim. Two functions are now
adapted: prompt tokenization and teacher-forced replay.

## Inspected alternatives and why they were not incorporated

| Candidate code inspected | Decision |
|---|---|
| `phase1_if_analysis/src/phase1/statistics.py` | Its bootstrap silently drops nonfinite samples and runs its own scalar resampling loop. KSRF requires explicit rejection and prompt-level paired/vectorized uncertainty; retain SciPy bootstrap instead. |
| `phase1_if_analysis/src/phase1/io.py` | Generic hash/atomic I/O is reusable, but the selected IMF excerpts supply these responsibilities without importing phase1 quantization/configuration code. |
| `phase1_if_analysis/src/phase1/reporting.py`, relevant `experiments.py` sections | Their summaries concern quantization drift, margins and fingerprint survival, not the KSRF derivative or sixteen-key controls. |
| `if-sft-turboquant-lab/scripts/run_vllm_if_fsr.py` and `protocol.py` | Greedy/top-k IF-SFT protocol and Vicuna text slicing/retokenization would violate Qwen sampling and exact saved-token replay. Existing full runner cannot be used unchanged. |
| `if-sft-turboquant-lab/scripts/state.py` | IF-specific completion schemas; no need to introduce an incompatible second manifest layer. |
| `if_awq_tier0/src/fingerprint_soft_metrics.py` | NLL/rank/margin implementation is not Gamma or a semantic-score derivative; it also tokenizes target text. Computing unrelated full-vocabulary rank/margin tensors adds cost. |
| `if_awq_tier0/src/common.py`, `phaseA_fingerprint/phasea/common.py` | Shared I/O/model helpers inspected. Prefer the selected atomic IMF excerpts; model helpers lack the required frozen revision protocol or enable unrelated TF32/performance behavior. |
| `imf_ptq/src/imf_ptq/model_io.py`, `calibration.py`, `provenance.py` | Loader rejects non-Llama models; calibration is C4 quantization data. Generic provenance overlaps the selected IMF hash helper. |
| `imf-tier0-ptq/src/imf_tier0/gpu/hf_runner.py` | HF text generation is prohibited for this experiment; per-prefix carrier caching is not full-trajectory replay. |
| `imf_ptq/.../stega/transformers_carrier.py` and its rtn2to4 copy | ADG carrier masks special/non-roundtripping tokens. That changes the required categorical distribution and removes EOS. |
| `tailor-q-guidedquant/tailor/stats.py`, `geometry.py`, `curvature.py`, `value/replay.py`, `value/rollout.py` | Grouped activation/quantization geometry and FVI replay/targets, not response trajectory Fisher or response-Jacobian fitting. Reusing them would alter the estimator. |
| `RBVT-squeeze/quantizers/sensitivity_store.py`, `squeezellm_collector.py` | Parameter-sensitivity storage/collection for quantization; not the 128D trajectory Fisher required here. |
| RIPPLE archive: `analysis/perturb_equivalence_analysis.py`, relevant functions in `analysis/mechanism.py`, `core/noise_in_hidden_test.py` | Teacher forcing and perturbation code inspected. The phase1 helper is a cleaner reusable causal replay implementation. RIPPLE perturbations are hidden/input/parameter Gaussian noise, not calibrated readout logit shifts or damped RMSNorm JVP inversion. Its generation is HF, not vLLM. |
| `rtn2to4/vendor/scalable/utils.py`, `MEraser/test_uft.py` | FSDP training storage / LoRA verifier and HF decoding are outside Experiment 1. |

No candidate implementing the same KSRF bridge, Haar key statistic, conditional
gradient test or native tangent inverse was identified in this audit. Those
remain custom experiment logic, with standard numerical operations delegated to
NumPy/SciPy/sklearn/PyTorch. This is a scoped finding, not proof about every file
or archive anywhere on the local disk.
