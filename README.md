# KSRF — Experiment 1

Complete implementation of [the supplied research specification](docs/KSRF_Experiment_1_Code_Spec.md):
clean-model profiling, held-out semantic response-Jacobian validation and independent
perturbed decoding. **This experiment does not train LoRA or implant a fingerprint.**

## Run the complete experiment on Google Colab

Use a Linux Colab GPU runtime with **bfloat16 support (L4/A100 or newer)** and an
existing, working **vLLM 0.20.0 / compatible Torch-CUDA** environment. The complete
dependency contract is [requirements.txt](requirements.txt). The script checks it
and reports missing or incompatible packages; it never installs anything, creates
an environment, or changes Torch/CUDA. Check available runtime/storage resources:
the run downloads Qwen 3B, BGE and UltraChat and generates thousands of responses.

Paste this single cell into Colab:

```bash
%%bash
set -euo pipefail
cd /content
if [ ! -d KSRF_fingerprint ]; then
  git clone https://github.com/HuyL13/KSRF_fingerprint.git
fi
cd /content/KSRF_fingerprint
bash run_colab.sh
```

`run_colab.sh` runs **every stage**, with the full 5,000 calibration / 1,500
held-out prompt selection. It includes mathematical checks, vLLM/HF distribution
parity, actual EOS/length-cap checks, clean generation, readout PCA, teacher-forced
trajectory scores, frozen BGE/PCA/whitening, bridge and ridge baseline fitting,
half-split/shuffled/random controls, sixteen keys and 2,000-replicate confidence
intervals, calibration-only delta selection, signed processor smoke checks,
independent finite differences, native RMSNorm tangent diagnostics and the report.

Finite differences start at 500 held-out prompts per eta and automatically extend
to all available 1,500 if positive trend and compatible uncertainty are not
established. Previous responses are retained during this extension. All response
generation uses batched offline `vllm.LLM.generate`; no HF generation fallback exists.

Each major GPU stage runs in a separate Python process so the vLLM engine, HF
replay model and BGE encoder do not compete for the same GPU memory. Use
`config.yaml` to adjust batching/VRAM utilization for the allocated GPU. Reducing
dataset counts is rejected by the full-run preflight.

Default output: `outputs/<UTC-run-timestamp>/`. To persist outputs on an **already
mounted** Drive, set `KSRF_OUTPUT_DIR` to a new directory before the command:

```bash
%%bash
set -euo pipefail
cd /content/KSRF_fingerprint
export KSRF_OUTPUT_DIR="/content/drive/MyDrive/KSRF/experiment-1-run-01"
bash run_colab.sh
```

The destination must be empty. A failed stage exits nonzero and records
`FAILED` in `pipeline_status.json`; no partial run is announced as complete.
`COMPLETE` means all computations and reporting finished, **not** that the
scientific hypothesis passed. A full GPU run has not been executed on the local
Windows CPU development machine; runtime parity is measured on your actual GPU
before collecting the experimental dataset.

## Outputs and interpretation

- `EXP1_REPORT.md`: revisions/runtime, stop and length statistics, mathematical
  checks, spectra/subspace stability, all sixteen held-out keys and controls,
  independent finite-difference confidence intervals/KL/examples, tangent
  residuals and an explicit `GO / INCONCLUSIVE / NO-GO` decision.
- `manifest.json`, `environment.json`, `resolved_config.json`: exact frozen Hub
  revisions, installed versions, seeds and runner configuration.
- `*_rollouts.jsonl`: actual prompt/output token IDs, sampled EOS metadata,
  per-request seeds and decoding settings. No full-vocabulary logits are saved.
- `*_features.npz`, `readout_features.npz`, `semantic_transform.npz`,
  `bridge.npz`, `bridge_halves.npz`, `keys.npz`: sufficient frozen numerical
  artifacts for refitting and inspection without repeating clean inference.
- `gradient_bootstrap.json`, `finite_difference.json`, `tangent_diagnostics.json`:
  the numerical validation results. `artifact_checksums.json` covers final files.

HF replay and vLLM use different GPU kernels. The parity stage records the
observed sparse processed-logprob errors and blocks the pipeline if the
predeclared tolerance is exceeded. Passing the check establishes a measured
numerical approximation, not bit-exact equality between kernels.

The report is conservative: even if every numerical gate passes, it returns
`INCONCLUSIVE` pending a researcher's inspection of the matched output examples
for degeneration. It does not invent an automatic fluency metric or call an
uncertain result a successful fingerprint. No post hoc key selection or
ownership-detection threshold is used.

After inspecting the actual matched outputs, a researcher can record
`quality_review.json` inside that run's directory, for example:

```json
{"reviewed": true, "reviewer": "Your name", "no_obvious_degeneration": true, "notes": "Describe the inspected examples."}
```

Rerun the `report` stage with the resolved configuration to incorporate that
judgment. `GO` is possible only when every numerical gate and that qualitative
review pass. This optional review does not stop the Bash pipeline from completing
all computations and producing its initial report.

## Inspect or rerun a stage

The Bash entrypoint invokes the full pipeline. Individual stage entrypoints are
also available for debugging, using a run's **resolved** configuration:

```bash
python -m ksrf_exp1 --help
python -m ksrf_exp1 gradient --config outputs/YOUR_RUN/resolved_config.json
python -m ksrf_exp1 report --config outputs/YOUR_RUN/resolved_config.json
```

Rerunning a stage is not automatic recovery of a failed full run; inspect its
prerequisite artifacts first. Start a fresh full run after changing parameters.

## Reuse and verification

See [LOCAL_CODE_REUSE.md](docs/LOCAL_CODE_REUSE.md) for actual local source code
incorporated into prompt handling, teacher-forced replay, atomic I/O and hashing,
and [UPSTREAM.md](docs/UPSTREAM.md) for library/API reuse. KSRF-specific Fisher/bridge/key/score logic is the main
custom code. PCA, ridge regression, bootstrap, root finding, principal angles,
iterative solves, embeddings and vLLM request movement reuse upstream libraries.

`python -m ksrf_exp1 mathematics --config config.yaml` runs the delivered
exact-enumeration mathematical diagnostic without model downloads or a GPU.
The full runner executes it before generating data. Development verification
also exercised real SciPy/sklearn synthetic fit/held-out analysis and CPU
JVP/VJP checks in pre-existing environments; it is not experimental evidence.
