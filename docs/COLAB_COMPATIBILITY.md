# Colab Latest compatibility and import audit

Audited on 2026-10-28. The target is **Latest**, as requested by the user. The
experiment Bash entrypoint starts its 14 experiment stages directly. It does
not call the separate `doctor`/`preflight` environment diagnostics, install
packages or replace the notebook's existing native runtime.

Update after the user's actual Colab traceback: the installed vLLM is **0.32.0**.
The old hardcoded 0.20.0 generation gate was incorrect for that runtime and has
been removed, including from the optional doctor. Existing resolved configs
containing the old field are ignored for library-version selection. The
requirements entry no longer pins vLLM. Exact versions are recorded, not enforced.
The 0.32.0 LLM/SamplingParams/AdapterLogitsProcessor/sampler source interfaces were
inspected directly; the API used by this project remains present and the custom
non-argmax-invariant processor still runs before temperature.

## Official snapshot compared

Source: [googlecolab/backend-info GPU package snapshot](https://github.com/googlecolab/backend-info/blob/aaf59a1929a8b637ef5d41272647c8bee3f8fe7c/pip-freeze.gpu.txt),
commit `aaf59a1929a8b637ef5d41272647c8bee3f8fe7c`.
Its OS snapshot records Python 3.13.16. These published files can lag a live
Colab runtime; they are evidence about that snapshot, not an inspection of the
user's connected notebook.

| Package | Latest GPU snapshot | Project compatibility |
|---|---|---|
| torch | 2.11.0+cu130 | Existing runtime; no requirements entry |
| torchvision | 0.26.0+cu130 | Existing runtime; no requirements entry |
| torchaudio | 2.11.0+cu130 | Existing runtime; no requirements entry |
| transformers | 5.20.2 | Accepted; previous `<5` bound was incorrect for Latest |
| huggingface-hub | 1.33.0 | Accepted; previous `<1` bound was incorrect for Latest |
| sentence-transformers | 5.7.0 | Accepted |
| datasets | 4.8.5 | Accepted |
| accelerate | 1.15.0 | Accepted |
| tokenizers | 0.23.3 | Matches Transformers 5.20.2 dependency metadata |
| numpy | 2.1.3 | Accepted |
| scipy | 1.16.3 | Accepted; CG supports rtol/atol |
| scikit-learn | 1.6.1 | Accepted |
| vllm | Not listed | Must already be available in the prepared runtime |

## Published dependency metadata

The [vLLM 0.20.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
uses PyTorch 2.11 and **CUDA 13.0 for its default wheel**. It also documents
Transformers v5 support. [PyPI's 0.20.0 metadata](https://pypi.org/pypi/vllm/0.20.0/json)
requires Python >=3.10,<3.15, torch/torchaudio 2.11.0 and torchvision 0.26.0.
Those public versions match the Latest snapshot's CUDA 13.0 variants. The
manylinux x86_64 wheel uses the cp38-abi3 tag, not a CPython-3.11-only tag.

The native packages are deliberately **not** added to `requirements.txt`.
The experiment runner performs no package installation. Matching dependency
metadata does not by itself prove that a particular installed wheel's native
extension loads and executes correctly.

Comparing the snapshot with [Transformers 5.20.2 metadata](https://pypi.org/pypi/transformers/5.20.2/json)
and [Sentence Transformers 5.7.0 metadata](https://pypi.org/pypi/sentence-transformers/5.7.0/json)
found no mandatory version gaps for those two packages. The project previously
rejected this stack because of overly restrictive Transformers/Hub bounds;
those bounds are corrected while retaining vLLM's published Transformer exclusions.

The stock snapshot is **not a ready vLLM environment**: vLLM itself is absent,
and comparing its mandatory dependency metadata identifies 34 missing or
incompatible entries (for example numba, lark and several decoding/kernel
packages). This audit does not install or repair those packages and does not
claim that a fresh Colab session can run the full experiment unprepared.

## Optional explicit diagnostic

```bash
python -m ksrf_exp1.doctor --output import_audit.json
```

This separate command:

1. Checks installed version metadata recursively, respecting dependency markers
   and active extras, rather than comparing only top-level versions.
2. Enumerates every source import, including imports inside functions, and
   imports the actual modules and named symbols. It also imports every project
   module and forces Qwen2/BERT model-class imports that Transformers otherwise
   resolves lazily.
3. Checks the constructor/call signatures used by the installed vLLM processor,
   sparse processed logprobs, SentenceTransformer encoding and SciPy CG.
4. Imports vLLM's compiled `_C` extension and executes its CUDA RMSNorm kernel
   on a small bfloat16 tensor. This can expose native-wheel/driver errors that
   metadata and Python imports alone miss.

The optional report's PASS is about these checks, not full-model decoding or a
completed research experiment. `run_colab.sh` does not invoke this command.
During clean generation the pipeline only records versions/device information
for the scientific report; recording that provenance is not an environment scan.

## Verification actually performed

- Three regression checks passed: the official Latest direct package versions
  are accepted, a deliberately broken import inside a function is detected, and
  dependency matching accepts a correct CUDA-local version while rejecting a
  different Torch public version.
- Real import checks ran in an existing CPU environment with Transformers
  5.17.0, Sentence Transformers 6.1.0 and Hub 1.33.0. The remaining failures were
  missing vLLM/datasets/native-domain packages and unavailable CUDA, not mistaken
  symbol names in the checked HF/SciPy/sklearn imports.
- Actual tiny `Qwen2ForCausalLM` teacher-forced replay, sampled-EOS trajectory
  score derivatives and actual `Qwen2RMSNorm` JVP/VJP/CG ran successfully.
- An actual local BERT/CLS SentenceTransformer encoder successfully exercised
  the same `encode` keyword arguments and normalized output contract. No model
  download was needed for these small architecture-level checks.
- Python compilation, Bash syntax and independent source/API review passed.

**Not executed:** Colab Latest itself, vLLM CUDA imports/kernel or full Qwen 3B
generation on an actual Colab GPU. Therefore this audit does not claim a full
GPU run passed or guarantee the absence of every possible runtime error.
