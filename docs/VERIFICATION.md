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
