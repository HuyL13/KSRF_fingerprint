# KSRF — Experiment 1: Validate Semantic Response Jacobian

**Version:** 1.1, 2026-10-28 (mandatory vLLM generation)  
**Status:** implementation-ready research specification; NOT evidence of success  
**Scope:** clean-model profiling, held-out validation, and a small independent generation test. **Do not train LoRA or implant the fingerprint in this experiment.**

## 0. Research question and decisions

Can a *key-independent, low-rank* linear operator estimated from clean-model trajectories reliably predict how a small **readout-space logit perturbation** changes the *full-response semantic embedding* on unseen prompts?

This experiment must resolve four questions, without guessing:

1. Is the estimated response Jacobian stable across independent calibration subsets?
2. Does it predict **held-out, prompt-conditioned, correct-key** semantic-score gain (not just the average gain over random keys)?
3. Does direct perturbed decoding confirm the sign/direction of the predicted gain?
4. Are the proposed logit shifts approximately attainable through the native pre-final-norm representation when the final normalization and LM head are frozen?

**Decision gate:** Only proceed to full teacher-oracle implantation and LoRA after these pass. An inconclusive statistical test warrants more held-out data, not necessarily a method redesign.

## 1. Environment and reproducibility

- Model: `Qwen/Qwen2.5-3B-Instruct` (record exact commit SHA).
- Encoder: `BAAI/bge-large-en-v1.5` (record exact revision; use the same frozen embedding pipeline for prompts and responses).
- Dataset: `HuggingFaceH4/ultrachat_200k` with `prompt` / `prompt_id`.
- Clean calibration prompts: 5,000 unique examples from `train_gen`.
- Held-out prompts: 1,500 unique examples from `test_gen`, **not used to fit any matrix, baseline, or hyperparameter**.
- Filter prompts to 16–512 tokens using the Qwen tokenizer. Log skipped prompts and any generated responses excluded for degenerate/empty decoded text.
- Prompt selection seed: `20261028`. Maintain separate seeds for model sampling (`731000 + index`), projection sampling (`1123`), and key generation (`4417`).
- Use `bfloat16` for model inference if necessary; all softmaxes, score accumulation, covariance estimation, eigendecompositions, and diagnostics in FP32/FP64.
- No online APIs, paraphrase model, DPO, SFT, or LoRA training in Experiment 1.
- **Mandatory generation backend: vLLM for ALL generated responses, including clean calibration/held-out rollouts and the `eta=-delta,0,+delta` validation rollouts.** Do not silently substitute `transformers.generate()` or a Python autoregressive sampling loop.

### Clean decoding protocol

```yaml
backend: vllm_offline_LLM_generate
model: Qwen/Qwen2.5-3B-Instruct
dtype: bfloat16
temperature: 0.8
top_p: 1.0
top_k: 0           # no top-k filtering (supported vLLM setting)
min_p: 0.0
repetition_penalty: 1.0
presence_penalty: 0.0
frequency_penalty: 0.0
max_tokens: 256
n: 1
ignore_eos: false
```

Use the exact same revision of Qwen and tokenizer for vLLM generation and Hugging Face teacher-forced replay. Tokenize with the normal Qwen chat template (`add_generation_prompt=True`) **once**, pass the resulting prompt token IDs to vLLM, and persist those exact IDs. Use `vllm.LLM.generate` in **batched offline inference**, with a reproducible `SamplingParams.seed` for each prompt (not a single shared batch seed). Pin and record the exact `vllm` package version, model/tokenizer revisions, runner configuration, and all sampling parameters. Disable any model-generation-config overrides, built-in watermarking, speculative decoding, custom stop strings, structured output, or unrequested logits processors; verify actual behavior in the installed version. Clean generation uses **no custom logits processor**.

**The intended categorical distribution is `softmax(raw_logits / 0.8)`.** vLLM and HF replay use different kernels, so numerical equality is not automatic. Before collecting the dataset, validate their tokenization, EOS behavior, and log-probability agreement on a small fixed set of contexts (using a supported vLLM log-probability diagnostic where practical). If the discrepancy is material, **do not claim the HF replay produces the exact score-function of the vLLM sampling distribution**; fix the mismatch or explicitly label the estimator approximate and block mathematical validation claims.

The output sequence used for trajectory scores includes any sampled EOS/`im_end` token. The response text sent to BGE excludes EOS and chat-control tokens. If `max_new_tokens` terminates a response, do *not* invent an EOS token in its score.

The distribution for generating a token and the distribution for computing its score-function contribution must be identical for an exact estimator; version/precision deviations must be measured and reported.

### vLLM generation implementation contract (mandatory)

For clean rollouts, use the **offline Python API** (`LLM`, `SamplingParams`), not remote/OpenAI APIs and not HF `generate()`. Run generation in batches, with one seed for each example. This is illustrative pseudocode; validate constructor arguments against the **pinned installed version**:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    dtype="bfloat16",
    tensor_parallel_size=1,  # raise only if allocated GPUs require it
    trust_remote_code=False,
)

prompts = [{"prompt_token_ids": make_qwen_chat_prompt_ids(x)} for x in texts]
params = [SamplingParams(
    temperature=0.8, top_p=1.0, top_k=0, min_p=0.0,
    repetition_penalty=1.0, presence_penalty=0.0,
    frequency_penalty=0.0, max_tokens=256,
    n=1, ignore_eos=False, seed=731000 + i,
) for i in example_indices]
outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
# Match results by stable request/prompt_id, not by unordered worker completion.
# Save outputs[i].outputs[0].token_ids and finish_reason/stop_reason.
```

Do not mix vLLM-generated **text** with independently re-tokenized response IDs. Save the actual generated token IDs, including any actual sampled terminal token if available. Test explicitly whether the installed vLLM returns sampled EOS/`im_end` in `output.token_ids`; **do not assume** it does. If a sampled stopping token is suppressed, recover it only from reliable engine metadata or a validated generation trace, and prove the reconstruction with a unit test; otherwise the sample cannot be used for an exact trajectory-score estimator. A length-cap completion must not be assigned a fictitious EOS.

**Perturbed generation in Section 6D must also use vLLM.** Implement a per-request, GPU-efficient custom logits processor using the API available in the pinned vLLM version. It must add the *full-vocabulary* bias for each prompt/key (no greenlist, top-token approximation, or CPU Python token loop). The desired distribution is:

```python
p_eta = softmax(raw_logits / tau + eta * (F_vocab @ w_K(x)))
```

vLLM custom logits processors in current interfaces act on **pre-temperature** logits. Therefore the processor must add:

```python
raw_logits += tau * eta * (F_vocab @ w_K(x))  # tau = 0.8
```

**NOT** `eta * (F_vocab @ w_K(x))` directly to raw logits, because that would yield a teacher strength scaled by `1/tau` after temperature. Cache the request's bias vector (or compact key coefficients with GPU projection) by request ID; support concurrent requests and `eta=-delta,0,+delta` without cross-request contamination. `eta=0` must be exactly a no-op. `SamplingParams.logit_bias` is acceptable only as a validated smoke-test fallback; do not serialize dense vocabulary-sized Python dictionaries for the main run. Pin the custom-processor API/runner version because vLLM has changed this interface.

Before Section 6D's main rollouts, perform a 2–5 prompt smoke test: (a) `eta=0` processor/no-processor give the same distribution; (b) inspect small-vocabulary or returned logprob diagnostics to confirm the **temperature scaling and sign** for both `+delta` and `-delta`; (c) verify request-specific bias and EOS/length stop handling. Do not fall back to HF generation if the custom processor is unsupported: report the blocker and fix the vLLM implementation first.

Use Hugging Face/PyTorch **only** for teacher-forced replay (full-vocabulary logits needed for `Gamma`), extraction of `lm_head`, PCA, Jacobian/Fisher computations, and RMSNorm/JVP diagnostics. vLLM is the mandatory runtime for **response generation**, not for all analytical forward passes.

## 2. Data generation and storage

**First pass:** generate exactly one clean response per calibration/held-out prompt **with batched vLLM offline inference**; store:

```json
{
  "prompt_id": "...",
  "prompt_text": "...",
  "prompt_token_ids": [1, 2],
  "generated_token_ids_including_sampled_eos": [3, 4],
  "response_text_without_special_tokens": "...",
  "stop_reason": "sampled_eos|length_cap",
  "sampling_seed": 731000,
  "generation_backend": "vllm",
  "vllm_version": "...",
  "model_revision": "...",
  "tokenizer_revision": "...",
  "sampling_params": {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 256}
}
```

Do **not** store all vocabulary logits. The saved token IDs allow exact teacher-forced replay and streaming accumulation. Use distinct files for calibration and held-out samples.

### Feature projection in native readout space

Let `W = model.get_output_embeddings().weight` with shape `[vocab_size, hidden_size]`. Use the actual `lm_head`, *not* an assumed tied input embedding. Compute a rank-128 orthonormal basis `P` from a seeded representative sample of 20,000 naturally sampled non-special response-token rows `W[token_id]`, using centered PCA/SVD. Fit `P` **only on calibration data** and save the centering mean and `P`.

For every vocabulary token `v`, define its frozen readout feature:

`f(v) = P.T @ W[v]`, dimension 128.

The shared PCA centering constant is unnecessary for the score-function difference or softmax logit shift because it cancels; retaining it in implementation is harmless if used consistently. Precompute `F_vocab = W @ P`, shape `[vocab, 128]`, in inference precision on GPU if memory permits.

**Second pass:** replay the exact **vLLM-sampled** prompt + sampled token IDs with the matching HF clean model under teacher forcing (not response generation). For every position that predicted a *sampled* output token (including sampled EOS), compute:

```python
p_t = softmax(raw_clean_logits_t.float() / 0.8, dim=-1)
gamma_t = F_vocab[sampled_token_t] - (p_t @ F_vocab)
Gamma_i = sum_t gamma_t  # shape [128], NOT divided by response length
```

Causal indexing: if prompt length is `L`, logits at positions `L - 1, L, ...` predict successive response tokens; avoid a one-token shift. Ignore all prompt prediction positions.

For BGE, compute deterministic normalized original embeddings `emb_prompt_i` and `emb_response_i`, and cache them. **Do not** truncate the decoded response beyond its encoder's configured limit without reporting that truncation; max 256 LLM tokens should help keep it within the BGE encoder's limit, but check explicitly.

Save `Gamma_i` in FP32/FP64, BGE prompt/response embeddings, sequence lengths, stop reason, token counts and checksums. This is sufficient to refit and validate without repeating clean LLM inference.

## 3. Semantic feature preprocessing

Fit all transformations **only on calibration embeddings**:

- Full frozen response BGE embedding -> center -> top-64 PCA projection: `phi(y) in R^64`. No need to renormalize *after* projection (it should remain an affine/linear map of the frozen semantic embedding).
- Full frozen prompt BGE embedding -> center -> top-8 PCA projection -> covariance regularization/whitening -> unit normalization: `z(x) in R^8`, `||z||=1`.
- Store all mean vectors, basis matrices, whitening parameters and encoder revisions. Never refit on held-out data.
- Keep BGE embedding preprocessing unchanged between original and paraphrased inputs in subsequent experiments.

Check for zero norms and nonfinite values; reject/report these examples without silent repair.

## 4. Fit the response–logit operator

For each calibration sample, `Gamma_i in R^128`, `phi_i in R^64`.

Use the calibration response mean `mu_phi`, and calculate:

```python
C = mean(outer(phi_i - mu_phi, Gamma_i))    # [64,128]
F = mean(outer(Gamma_i, Gamma_i))           # [128,128]
F = (F + F.T) / 2
F_reg = F + 0.05 * trace(F) / 128 * I
L = inv_sqrt_symmetric(F_reg)
B = C @ L                                  # [64,128]
U, singular_values, Vh = svd(B, full_matrices=False)
U8 = U[:, :8]
V8 = Vh.T[:, :8]
```

`C` is the **population derivative estimator** corresponding to a *constant readout-feature logit shift* when the score-function identity is implemented exactly. `F` is the trajectory Fisher matrix in those same coordinates, estimated from clean trajectories; shrinkage avoids unstable inverse square roots.

Store all matrices, eigenvalues, condition number, singular values, exact dataset split and random seeds. Verify `U8.T @ U8 ≈ I`, `V8.T @ V8 ≈ I`, and numerical finiteness.

**Critical limitation:** a global `C = E_x[J(x)]` does **not** imply that the semantic-score derivative is nonnegative for each prompt or even for a particular secret key. Never use `b(x)^T Sigma b(x) >= 0` as a per-prompt or fixed-key guarantee.

## 5. Define independent keys and the diagnostic score

For each key `K`, sample an `8 x 8` Haar orthogonal matrix `R_K` via QR of a fresh Gaussian matrix and the standard diagonal-sign correction. Do not force determinant `+1`; the full orthogonal distribution gives sign symmetry.

```python
b_K(x) = R_K @ z(x)                    # [8]
w_K(x) = L @ V8 @ b_K(x)             # [128]
a_K(x) = U8 @ b_K(x)                 # [64]
```

The per-prompt logit shift to be implemented in a subsequent teacher experiment is:

`delta_logits_v = eta * (F_vocab[v] @ w_K(x))`.

Note `eta` multiplies the scores *after* dividing clean logits by sampling temperature (i.e. `softmax(raw_logits / 0.8 + eta * delta_logits_feature)`). Keep this convention fixed.

### Key-independent prompt–response baseline (recommended)

Natural prompt–response semantic alignment can dominate a small fingerprint signal. Fit a **small ridge regression** from the calibration prompt embedding (plus an intercept) to `U8.T @ phi(y)` (8D). Call its prediction `g(x)`.

- Use `g(x)` only for residualization of semantic scores.
- The ridge model depends on neither secret key nor suspect model.
- This is **not** a separate response classifier, an empirical ownership threshold, or a clean-model response generated at verification time.
- Do not tune regularization using fingerprint-detection results. Use a fixed ridge parameter or plain internal calibration cross-validation of response-prediction MSE.
- Save both residualized and non-residualized scores for ablation.

Score for a prompt/response batch `Q`:

```python
T(K, Q) = mean_i dot(b_K(x_i), U8.T @ phi(y_i) - g(x_i))
```

For the score-function derivative, the baseline `g(x)` is a constant conditional on `x` and therefore cannot change the true derivative (though it can lower variance).

## 6. Validate without training a model

### 6A. Mathematical/implementation unit tests (must pass first)

1. Construct a tiny toy autoregressive model (3–5 vocabulary tokens; max 2–3 generated tokens) with a deterministic table mapping full token sequences to vector-valued `phi(y)`. Enumerate **all** trajectories exactly. Check `dE[phi(Y)]/deta|0 == E[phi(Y) Gamma.T]` against central finite differences within numerical tolerance.
2. Verify per-position `E_{v~p_t}[f(v)-E_{p_t}f] ≈ 0` (use an exact vocabulary sum, not sampled tokens).
3. Verify the score of all sampled tokens equals the derivative of their **full trajectory log-probability**, including EOS where sampled.
4. Verify temperature scaling, causal index, chat template, and no unintended generation processor; compare HF replay vs vLLM logprobs on fixed contexts.
5. Verify vLLM token IDs, sampled EOS/`im_end`, `finish_reason`, per-request seeds, and deterministic `prompt_id` alignment (including two different batch sizes). Test an `eta=0` custom-processor no-op and its `tau*eta` pre-temperature correction.
6. Under a fixed `x`, the expected score contribution of the key-independent baseline `g(x)` is zero.
7. Use **vLLM** to generate a sequence with length cap and one with sampled EOS to test stop rules.

### 6B. Subspace stability (calibration split)

Using the frozen `P` and semantic preprocessing, fit `(C,F,U8,V8)` separately on the first and second halves of calibration records. Compare:

- leading singular value spectra;
- principal angles between `span(U8)` subspaces;
- principal angles between `span(L @ V8)` native readout spaces;
- held-out projected `C` alignment / normalized singular trace;
- shuffle control: permute response `phi` across prompts while holding `Gamma` fixed, refit `C`, and compare singular strengths against the true pairing.

Do not compare column-by-column singular vectors because individual vectors are sign/rotation ambiguous.

### 6C. Direct held-out *key-conditioned* derivative test (primary)

Use **only** the held-out prompt-response records, never samples used for fitting.

For 16 independently generated keys (one predeclared primary key and 15 diagnostic keys), estimate:

```python
r_i(K) = dot(b_K(x_i), U8.T @ phi_i - g(x_i))
d_i(K) = dot(Gamma_i, w_K(x_i))
Ghat_K = mean_i r_i(K) * d_i(K)
```

`Ghat_K` is the score-function estimate of the derivative of the **expected verifier statistic** with respect to `eta`, at `eta=0`, for that key. It directly catches the mistake of replacing prompt-dependent `J(x)` by global `C`.

Use bootstrap over complete prompt-response pairs (e.g. 2,000 replicates) to obtain a 95% confidence interval for `Ghat_K`. Report all 16 key values; **never select the best key after inspecting held-out scores**. Compare against:

- keys with randomly permuted prompt–response pairing;
- an `r=8` random semantic/readout subspace baseline, with matched native Fisher norm;
- the same score without the residualization baseline `g(x)`.

Useful acceptance evidence: the predeclared primary key has a **positive 95% lower confidence bound**; most diagnostic keys have positive derivatives; the true subspace clearly outperforms permutation/random controls. Inconclusive confidence intervals warrant more held-out data; they do not by themselves prove failure.

Mathematical cross-check: if `||z(x)||=1` and `R_K` is uniform on `O(8)`, then for population quantities:

`E_K [G_K] = trace(U8.T @ C @ L @ V8) / 8 = sum_j singular_values[j] / 8 >= 0`.

**This is a key-averaged result, NOT a per-key guarantee.** With finite samples this is only approximate.

### 6D. Independent perturb-and-generate confirmation (secondary)

This is a small, **independent** perturbed-decoding smoke test using held-out prompts, not a fingerprint implantation. **All `-delta`, `0`, and `+delta` responses MUST be generated with batched vLLM using the validated per-request custom logits processor described in Section 1.** No HF `generate()` or handwritten sampling loop is permitted.

For the predeclared primary key, sample response datasets using:

- `eta = 0` (clean);
- `eta = +delta`;
- `eta = -delta`.

Use the **same decoding temperature/protocol** and `softmax(raw_logits / 0.8 + eta * F_vocab @ w_K(x))`. Choose `delta` on calibration contexts **before seeing held-out response scores**, so the mean token KL at `+delta` versus clean is initially about `0.002` nats (also report p99 token KL). This is a finite perturbation; it will only approximate the derivative near zero. No top-k or top-p truncation.

Start with 500 held-out prompts; grow toward 1,500 if the result is statistically underpowered. Record output quality examples, score means and confidence intervals. Report:

```text
D_fd = (mean T(+delta) - mean T(-delta)) / (2*delta)
D_score = Ghat_K from independent clean held-out trajectories
```

Look for consistent signs and compatible uncertainty intervals. **Do not require exact numerical equality**, because `delta` is finite and sampling noise may dominate. Independent generation is essential: merely recomputing the same score-function estimator would not validate the practical teacher effect.

### 6E. Native-readout realizability sanity check

For a fixed set of held-out clean pre-final-norm states `r_t`, let `N` be the model's actual final RMSNorm and `W` its frozen LM head. The desired logit shift is `eta * W @ P @ w_K(x)`.

Estimate the Jacobian–vector map `D N(r_t) [delta_r]` numerically or via JVP at representative states. Given target post-norm change `P @ w_K(x)`, project it onto the local tangent image of `D N(r_t)` (or solve a damped least-squares inverse using JVP/VJP without materializing the full 2048x2048 Jacobian). Measure:

`relative_tangent_residual = ||target - achievable_tangent|| / ||target||`, and `required_residual_shift_norm = ||delta_r|| / ||r_t||`.

For RMSNorm, the radial direction is strongly attenuated: with a strictly positive RMSNorm epsilon the Jacobian can be full-rank mathematically, but its radial direction may be extremely ill-conditioned. Use a *damped* inverse or singular-value cutoff, and report both the residual and the norm of the pre-norm perturbation required. Do not falsely treat its image as exactly codimension one in finite-epsilon arithmetic. No LoRA training or model-weight changes are necessary.

This is a *necessary sanity check*, **not proof** that LoRA can realize the desired shift at all contexts. Report median/p95 residual; a consistently large residual is a warning.

## 7. Acceptance and failure interpretation

**GO to full teacher-oracle:**

- All mathematical score-function, temperature, indexing and EOS tests pass; vLLM clean/perturbed generation and HF replay distribution cross-checks pass.
- True Jacobian subspace is demonstrably more stable/informative than shuffled/random baselines.
- Predeclared primary key has positive held-out `Ghat` with useful uncertainty; key-averaged results are positive.
- Perturbed decoding shows the expected positive semantic-score trend without obvious output degeneration; if underpowered, increase held-out data rather than declaring success.
- Tangent realizability residual is not pathologically large.

**NO-GO / redesign trigger:** systematic negative or near-zero held-out key-conditioned derivatives **despite adequate sample size**, unstable subspaces, no independent decoding trend despite a strong predicted gradient, or consistently unattainable native readout shifts. Diagnose each separately.

**Do not claim yet:** paraphrase robustness, ownership TPR, fixed-key FPR, adaptive removal resistance, quantization robustness or LoRA learnability. Those are subsequent experiments.

## 8. Suggested code layout

```text
ksrf_exp1/
  config.yaml
  data.py                   # deterministic splits and chat prompts
  clean_rollouts.py         # batched vLLM: sample token sequences + EOS metadata
  vllm_logit_processor.py  # batched request-specific +delta/-delta raw-logit biases
  validate_vllm_parity.py  # vLLM/HF distributions, EOS, logit scaling tests
  readout_features.py       # PCA of sampled W rows, f(v)
  replay_scores.py          # teacher-forced Gamma and checks
  semantic_features.py      # frozen BGE embeddings; PCA/whitening
  fit_bridge.py             # C, F, shrinkage, SVD, ridge baseline
  key_and_statistic.py      # Haar R, b(x), w(x), T(K,Q)
  validate_gradient.py      # held-out Ghat + confidence intervals
  validate_fd.py            # perturbed independent decoding
  validate_tangent.py       # RMSNorm local JVP realizability
  tests/test_score_identity.py
  tests/test_sampling_indexing.py
  tests/test_vllm_generation.py
  tests/test_vllm_logits_bias.py
  tests/test_eos_and_length.py
  report.py
  outputs/
    manifest.json
    calibration_rollouts.jsonl
    heldout_rollouts.jsonl
    calibration_features.npz
    heldout_features.npz
    bridge.npz
    vllm_parity.json
    key_diagnostics.json
    gradient_bootstrap.json
    finite_difference.json
    tangent_diagnostics.json
    EXP1_REPORT.md
```

### Minimal conceptual pseudocode

```python
# Calibration (clean, frozen)
W = clean_model.get_output_embeddings().weight
P = fit_pca_on_sampled_readout_rows(W, sampled_clean_token_ids, k=128)
F_vocab = (W.float() @ P.float())  # can chunk by vocab to control memory

for x, sampled_y_tokens, response_text in calibration_rollouts:
    Gamma = zeros(128)
    for raw_logits, sampled_token in replay_response_positions(x, sampled_y_tokens):
        p = softmax(raw_logits.float() / 0.8, dim=-1)
        Gamma += F_vocab[sampled_token] - (p @ F_vocab)
    save(Gamma, bge_prompt(x), bge_response(response_text))

Phi, Z = fit_frozen_semantic_projections(calibration_bge)
C = mean_outer(Phi - Phi.mean(0), Gamma)
F = mean_outer(Gamma, Gamma)
L = inv_sqrt(F + 0.05 * trace(F) / 128 * eye(128))
U, Sigma, Vt = svd(C @ L)
U8, V8 = U[:, :8], Vt.T[:, :8]
g = fit_ridge(prompt_features, Phi @ U8)

# Independent held-out test (no parameter updates)
R_primary = random_haar_orthogonal_matrix(8, key_seed=4417)
for x, y, Gamma, Phi, z in heldout_records:
    b = R_primary @ z
    w = L @ V8 @ b
    r = dot(b, U8.T @ Phi - g(x))
    d = dot(Gamma, w)
    per_prompt_contribution.append(r * d)
Ghat = mean(per_prompt_contribution)
ci = bootstrap_by_prompt(per_prompt_contribution)
```

## 9. Final output contract

`EXP1_REPORT.md` must have:

1. Dataset, vLLM version/runner/processor version, exact checkpoint and tokenizer revisions, numbers of EOS/length-cap sequences, mean/p95 lengths, clean/perturbed generation parameters, and GPU batching settings.
2. Score-function unit-test PASS/FAIL including EOS and temperature; report vLLM-to-HF replay parity and verified pre-temperature logits-processor scaling.
3. `C`, `F`, singular spectra, subspace angles, shuffled/random-control comparisons.
4. Main 16-key held-out `Ghat` table with CI (predeclared key marked).
5. Finite-difference scores at `-delta`, `0`, `+delta` including CI, token KL and example quality.
6. Tangent realizability median/p95.
7. Explicit `GO / INCONCLUSIVE / NO-GO` with the exact reason.
8. No post hoc key selection, no changed verifier thresholds, and no claims beyond this feasibility experiment.

## 10. Source checks

- `HuggingFaceH4/ultrachat_200k` has `train_gen` / `test_gen`, `prompt`, and `prompt_id`: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- `Qwen/Qwen2.5-3B-Instruct` architecture and original model card: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- vLLM `SamplingParams` (temperature, seed, token IDs, logit bias): https://docs.vllm.ai/en/latest/api/vllm/sampling_params/
- vLLM custom logits processors (pin the installed runner version): https://docs.vllm.ai/en/latest/features/custom_logitsprocs/
- Score-function identity background: https://wanghemath.github.io/Book-MathRL/chapters/chapter-15-policy-gradient-methods.html
- Related fingerprint threat-model work: https://arxiv.org/abs/2509.26598
