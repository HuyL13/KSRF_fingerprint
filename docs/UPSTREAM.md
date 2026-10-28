# Upstream reuse

The implementation was based on the supplied Experiment 1 specification, with
the user's explicit constraints: a new folder, a full Colab Bash runner, no
environment/Torch changes and maximum appropriate upstream reuse.

## Local source inspected

`D:/Downloads/if-sft-turboquant-lab/upstream/vllm` was already available locally:

- Repository: https://github.com/vllm-project/vllm
- Tag: `v0.20.0`
- Commit: `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`
- Files inspected: `vllm/v1/sample/logits_processor/__init__.py`,
  `interface.py`, `vllm/v1/sample/sampler.py`, `vllm/sampling_params.py`,
  `vllm/entrypoints/llm.py`, `docs/features/custom_logitsprocs.md` and
  `requirements/common.txt`.

The project imports the installed vLLM adapter directly rather than copying
its scheduler. `AdapterLogitsProcessor` owns added/removed/moved request state,
including continuous batching. The custom layer supplies only a full-vocabulary
GPU bias vector and an exact eta-zero bypass. This is why the vLLM version is
pinned and checked. The upstream sampler applies non-argmax-invariant custom
processors before temperature, so the raw correction is `temperature * eta * bias`.

Existing local fingerprint projects (`if_awq_tier0`, `phaseA_fingerprint`,
`LLM-Fingerprint-and-Attacks`, `MEraser`) were searched for reusable semantic
bridge/statistical components. Their available fingerprint scoring/quantization
logic did not implement this experiment's estimator; importing those heuristics
would change the research question, so standard numerical libraries are reused.

## Reused implementation components

| Component | Upstream implementation |
|---|---|
| Batched seeded clean/perturbed decoding | `vllm.LLM`, `vllm.SamplingParams` |
| Request add/remove/move bookkeeping | `vllm.v1.sample.logits_processor.AdapterLogitsProcessor` |
| Qwen chat template, checkpoint and teacher-forced logits | Transformers `AutoTokenizer`, `AutoModelForCausalLM` |
| Frozen normalized BGE embeddings | `sentence_transformers.SentenceTransformer.encode` |
| Dataset loading and deterministic shuffling | Hugging Face `datasets` |
| Exact model/encoder/dataset commit resolution | `huggingface_hub.HfApi.repo_info` |
| Semantic PCA and fixed ridge baseline | sklearn `PCA`, `Ridge` |
| Fisher eigendecomposition, bridge SVD and Haar QR | NumPy linear algebra |
| Prompt-level confidence intervals | `scipy.stats.bootstrap` |
| Principal subspace angles | `scipy.linalg.subspace_angles` |
| Calibration-only KL root solve | `scipy.optimize.brentq` |
| Actual final RMSNorm derivatives | `torch.func.jvp`, `torch.func.vjp` |
| Damped matrix-free inverse | `scipy.sparse.linalg.LinearOperator`, `cg` |

Reference documentation:

- https://docs.vllm.ai/en/latest/features/custom_logitsprocs/
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- https://huggingface.co/BAAI/bge-large-en-v1.5
- https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Dependencies retain their upstream licenses. No upstream model weights or whole
third-party repositories are committed here. Mathematical diagnostics enumerate
a tiny finite model only; that code is never used for experimental generation.
