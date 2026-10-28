"""Mandatory offline batched vLLM decoding; no HF generation path."""
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
import numpy as np
from .common import output_dir, read_jsonl, write_json, write_jsonl
from .data import frozen_config

def generate(cfg, records, etas=None, weights=None, processor=False, seed_offset=0, diagnostic=False):
    os.environ['VLLM_PLUGINS'] = ''
    os.environ['VLLM_LOGITS_PROCESSOR_PLUGINS'] = ''
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    cfg = frozen_config(cfg)
    version = importlib.metadata.version('vllm')
    if version != cfg['vllm_version']:
        raise RuntimeError('Expected pinned vLLM ' + cfg['vllm_version'] + ', found ' + version)
    tok = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['tokenizer_revision'])
    config = AutoConfig.from_pretrained(cfg['model'], revision=cfg['model_revision'])
    generation_config = GenerationConfig.from_pretrained(cfg['model'], revision=cfg['model_revision'])
    eos_ids = set()
    for value in [tok.eos_token_id, config.eos_token_id, generation_config.eos_token_id]:
        if value is not None:
            eos_ids.update(value if isinstance(value, list) else [value])
    kwargs = dict(model=cfg['model'],
        tokenizer=cfg['model'],
        revision=cfg['model_revision'],
        tokenizer_revision=cfg['tokenizer_revision'],
        dtype='bfloat16',
        tensor_parallel_size=cfg['tensor_parallel_size'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'],
        max_model_len=cfg['max_model_len'],
        generation_config='vllm',
        trust_remote_code=False,
        seed=cfg['sampling_seed'],
        logprobs_mode='processed_logprobs',
        enable_prefix_caching=False)
    if processor:
        kwargs['logits_processors'] = ['ksrf_exp1.vllm_logit_processor:KSRFLogitsProcessor']
    llm = LLM(**kwargs)
    result = []
    try:
        for start in range(0, len(records), cfg['generation_batch_size']):
            batch = records[start:start + cfg['generation_batch_size']]
            params = []
            for j, r in enumerate(batch, start):
                kw = dict(temperature=cfg['temperature'],
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    max_tokens=cfg['max_tokens'],
                    n=1,
                    ignore_eos=False,
                    seed=int(r['sampling_seed']) + seed_offset,
                    skip_special_tokens=False,
                    stop_token_ids=sorted(eos_ids))
                if diagnostic:
                    kw.update(logprobs=20, logprob_token_ids=cfg.get('diagnostic_token_ids'))
                if processor:
                    kw['extra_args'] = {'ksrf_eta': float(etas[j] if etas is not None else 0.0),
                        'ksrf_weights': np.asarray(weights[j]).tolist() if weights is not None else [],
                        'ksrf_features': str(Path(cfg.get('ksrf_feature_path',
                        output_dir(cfg) / 'readout_features.npz')).resolve())}
                if cfg.get('_parity_diagnostic', False):
                    kw.update(r.get('diagnostic_sampling_params', {}))
                params.append(SamplingParams(**kw))
            outputs = llm.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in batch],
                sampling_params=params,
                use_tqdm=True)
            if len(outputs) != len(batch):
                raise RuntimeError('vLLM dropped request')
            for r, o, p in zip(batch, outputs, params):
                if list(o.prompt_token_ids) != r['prompt_token_ids']:
                    raise RuntimeError('Request/tokenization mismatch')
                s = o.outputs[0]
                ids = list(s.token_ids)
                stop = s.stop_reason
                if s.finish_reason == 'length':
                    reason = 'length_cap'
                elif s.finish_reason == 'stop':
                    reason = 'sampled_eos'
                    if not ids or ids[-1] not in eos_ids or (isinstance(stop, int) and ids[-1] != stop):
                        raise RuntimeError('Sampled EOS absent from returned IDs; exact estimator blocked')
                else:
                    raise RuntimeError('Unexpected stop: ' + str(s.finish_reason))
                row = dict(r,
                    generated_token_ids_including_sampled_eos=ids,
                    response_text_without_special_tokens=tok.decode(ids,
                    skip_special_tokens=True),
                    stop_reason=reason,
                    engine_stop_reason=stop,
                    finish_reason=s.finish_reason,
                    sampling_seed=int(p.seed),
                    generation_backend='vllm',
                    engine_request_id=str(o.request_id),
                    vllm_version=version,
                    model_revision=cfg['model_revision'],
                    tokenizer_revision=cfg['tokenizer_revision'],
                    eos_token_ids=sorted(eos_ids),
                    sampling_params={k: getattr(p,
                    k) for k in ['temperature',
                    'top_p',
                    'top_k',
                    'min_p',
                    'max_tokens',
                    'ignore_eos',
                    'stop_token_ids',
                    'repetition_penalty',
                    'presence_penalty',
                    'frequency_penalty',
                    'n', 'seed', 'stop', 'min_tokens', 'skip_special_tokens',
                    'include_stop_str_in_output', 'logit_bias', 'allowed_token_ids']})
                if diagnostic:
                    row['logprobs'] = [{str(k): float(v.logprob) for k, v in step.items()} for step in s.logprobs]
                result.append(row)
    finally:
        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    return result

def run(cfg):
    import torch
    splits = ['calibration', 'heldout']
    prompts = {split: read_jsonl(output_dir(cfg) / (split + '_prompts.jsonl')) for split in splits}
    combined = generate(cfg, prompts['calibration'] + prompts['heldout'])
    # Record the runtime actually used for the experiment; this is provenance,
    # not a dependency scan or a gate before generation.
    package_versions = {}
    for name in ['torch', 'vllm', 'transformers', 'sentence-transformers', 'datasets',
                 'numpy', 'scipy', 'scikit-learn', 'huggingface-hub']:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    write_json(output_dir(cfg) / 'environment.json', {
        'python': sys.version, 'platform': platform.platform(), 'versions': package_versions,
        'cuda': torch.version.cuda,
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'processor_api': 'vllm.v1.sample.logits_processor.AdapterLogitsProcessor',
        'installation_performed': False,
        'record_type': 'runtime provenance after actual clean generation; no environment audit was run'})
    boundary = len(prompts['calibration'])
    for split, rows in [('calibration', combined[:boundary]), ('heldout', combined[boundary:])]:
        rejected = [{'prompt_id': r['prompt_id'],
            'reason': 'empty_response'} for r in rows if not r['response_text_without_special_tokens'].strip()]
        write_jsonl(output_dir(cfg) / (split + '_excluded.jsonl'), rejected)
        write_jsonl(output_dir(cfg) / (split + '_rollouts.jsonl'), rows)
