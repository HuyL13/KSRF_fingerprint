# Development verification

Verified on 2026-10-28 in existing Windows CPU environments. No dependency was
installed, removed or upgraded, and no Torch/CUDA environment was modified.

| Check | Result |
|---|---|
| Compile all Python modules | PASS |
| Bash syntax (`bash -n run_colab.sh`) | PASS |
| CLI stage discovery/help | PASS |
| Exact finite autoregressive enumeration, sampled EOS and length caps | PASS |
| Full-trajectory score-function derivative and zero-mean baseline | PASS |
| Actual replay helper causal indexing and score at temperature 0.8 | PASS |
| Fisher whitening, bridge basis orthogonality and Haar key properties | PASS |
| Real sklearn Ridge + SciPy bootstrap synthetic fit/held-out analysis | PASS |
| Frozen transforms and finite-difference analyzer on known independent branches | PASS |
| Decision gates: quality review, negative primary key, failed tangent solve | PASS |
| Actual torch.func JVP/VJP + SciPy CG compared with tiny dense reference | PASS |
| Full-vocabulary processor sign/temperature/zero bypass in CPU adapter harness | PASS |
| Bash full entrypoint on unsupported local environment | Correct nonzero exit; preflight marked FAILED |
| Independent source/API review, including follow-up on both findings | No blocking findings after fixes |

The local default interpreter had NumPy 1.26.4 and Torch 2.5.1 CPU. An already
available interpreter supplied SciPy 1.17.1, sklearn 1.9.1, NumPy 2.4.6 and
Torch 2.14 CPU for numerical integration checks. These synthetic checks are not
experimental evidence and were never written into an Experiment 1 report.

**Not executed locally:** real vLLM CUDA decoding, Qwen/HF model parity, BGE
encoding, the full UltraChat dataset run, actual fitted-readout perturbations or
full-model tangent diagnostics. Those are mandatory stages of `run_colab.sh`,
not skipped checks or precomputed PASS artifacts. A complete GPU run is required
before making experimental claims.

Temporary implementation test files were removed after execution. The delivered
`ksrf_exp1.mathematics` module remains an executable prerequisite of every full
run; it is not a replacement for the real-model GPU checks.

Follow-up local source reuse was checked with three regression cases: rejecting
non-object JSONL rows, preserving an existing artifact without leftover temporary
files after failed serialization, and rejecting teacher-forced context overflow
before forwarding the model. All three failed against the previous implementation
and passed after integrating the local code. The delivered mathematical/replay
diagnostic also passes after the integration. Five reused function bodies were
compared against their original source and match exactly; the one adapted
teacher-forcing function is explicitly documented in `LOCAL_CODE_REUSE.md`.

Latest-runtime follow-up: the old Transformers `<5` and Hub `<1` bounds rejected
the official Latest snapshot and were corrected. The standalone `doctor` command
now checks actual imports, transitive requirements, API signatures and a native
vLLM CUDA kernel when available. It is explicitly excluded from the Bash runner
at the user's request; the runner now has 14 experiment stages. Runtime metadata
needed by the report is recorded during actual clean generation instead.
Architecture-level CPU checks on actual Qwen2/BERT implementations passed. See
`COLAB_COMPATIBILITY.md` for the exact stack and verification limits.

Follow-up to the actual Colab vLLM 0.32.0 traceback: removed the hardcoded
0.20.0 version gate from generation and the optional doctor. A regression
fixture with runtime version 0.32.0 and an old resolved config containing 0.20.0
reproduced the reported failure before the fix, then reached engine generation
and recorded 0.32.0 after the fix. This fixture verifies gate removal, not real
GPU execution. The published 0.32.0 LLM, SamplingParams, EngineArgs, adapter and
sampler interfaces used here were inspected; no package was installed or changed.

vLLM 0.32 sparse-logprob follow-up: its actual `_validate_logprobs` implementation
rejects `logprobs=20` paired with five specific token IDs. The diagnostic request
now sets the count to the requested list length, covering both parity and fitted
readout smoke tests. The unchanged upstream validator rejected the old request
and accepted the fixed five-ID, six-ID and ordinary top-logprob requests in an
isolated CPU invocation. This proves request-validation compatibility, not a
completed GPU parity run. No dependency or environment change was made.
