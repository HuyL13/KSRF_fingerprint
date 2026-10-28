"""Isolated GPU parity diagnostics; a failure blocks exact-score claims."""
import argparse
import subprocess
import sys
import numpy as np
from .common import load_npz, output_dir, read_json, read_jsonl, save_npz, write_json, write_jsonl
from .data import frozen_config

def _child(cfg, mode):
    out = output_dir(cfg)
    if mode.startswith('smoke-'):
        return _smoke_child(cfg, mode)
    if mode == 'hf':
        import torch
        from .replay_scores import load_model, response_logits
        model = load_model(cfg)
        errors = {}
        with np.load(out / 'parity_features.npz') as f:
            features = f['F_vocab'][:, 0]
        for name in ['clean', 'zero', 'plus', 'minus', 'single', 'eos', 'cap']:
            rows = read_jsonl(out / ('parity_' + name + '.jsonl'))
            err = []
            for row in rows:
                logits = response_logits(model, row) / cfg['temperature']
                eta = row.get('_eta', 0.0)
                weight = row.get('_weight', 1.0)
                if eta:
                    logits += torch.as_tensor(features * eta * weight, device=logits.device)[None, :]
                if name == 'eos':
                    logits[:, int(row['_forced_eos'])] += 100.0 / cfg['temperature']
                lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
                for t, step in enumerate(row['logprobs']):
                    err.extend((abs(float(lp[t, int(k)]) - v) for k, v in step.items()))
            errors[name] = {'max_abs_logprob_error': float(max(err)),
                'mean_abs_logprob_error': float(np.mean(err)),
                'comparisons': len(err)}
        write_json(out / 'parity_hf.json', errors)
        return
    from .clean_rollouts import generate
    rows = read_jsonl(out / 'parity_contexts.jsonl')
    processor = mode in ['zero', 'plus', 'minus']
    eta = {'zero': 0.0, 'plus': 0.07, 'minus': -0.07}.get(mode, 0.0)
    if mode == 'single':
        cfg = dict(cfg, generation_batch_size=1)
    if mode == 'cap':
        rows = [dict(rows[0], diagnostic_sampling_params={'max_tokens': 1})]
    if mode == 'eos':
        rows = [dict(rows[0], _forced_eos=int(token_id), diagnostic_sampling_params={'logit_bias': {int(token_id): 100.0}}) for token_id in cfg['_eos_token_ids']]
    weights = np.asarray([[1.0 if j % 2 == 0 else -1.0] for j in range(len(rows))])
    generated = generate(cfg, rows, etas=np.full(len(rows), eta), weights=weights, processor=processor, diagnostic=True)
    for row, w in zip(generated, weights):
        row.update(_eta=eta, _weight=float(w[0]))
    write_jsonl(out / ('parity_' + mode + '.jsonl'), generated)

def run(cfg):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    cfg = frozen_config(cfg)
    out = output_dir(cfg)
    cfg = dict(cfg, max_tokens=cfg['parity_tokens'], _parity_diagnostic=True)
    tok = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['tokenizer_revision'])
    cfg['_eos_token_id'] = tok.eos_token_id
    model_config=AutoConfig.from_pretrained(cfg['model'],revision=cfg['model_revision'])
    generation_config=GenerationConfig.from_pretrained(cfg['model'],revision=cfg['model_revision'])
    eos_ids={tok.eos_token_id}
    for value in [model_config.eos_token_id,generation_config.eos_token_id]:
        eos_ids.update(value if isinstance(value,list) else [value])
    cfg['_eos_token_ids']=sorted(eos_ids)
    cfg['diagnostic_token_ids'] = [0, 1, 2, 100, tok.eos_token_id]
    vocab = AutoConfig.from_pretrained(cfg['model'], revision=cfg['model_revision']).vocab_size
    save_npz(out / 'parity_features.npz', F_vocab=np.sin(np.arange(vocab, dtype=np.float32) * 0.013)[:, None])
    cfg['ksrf_feature_path'] = str((out / 'parity_features.npz').resolve())
    rows = read_jsonl(out / 'calibration_prompts.jsonl')[:cfg['parity_prompts']]
    if len(rows) < 2:
        raise ValueError('Parity requires at least two prompts')
    write_jsonl(out / 'parity_contexts.jsonl', rows)
    config_path = out / 'parity_config.json'
    write_json(config_path, cfg)
    try:
        for mode in ['clean', 'zero', 'plus', 'minus', 'single', 'eos', 'cap', 'hf']:
            subprocess.run([sys.executable,
                '-m',
                'ksrf_exp1.validate_vllm_parity',
                '--config',
                str(config_path),
                '--worker',
                mode],
                check=True)
        error = read_json(out / 'parity_hf.json')
        clean = read_jsonl(out / 'parity_clean.jsonl')
        zero = read_jsonl(out / 'parity_zero.jsonl')
        single = read_jsonl(out / 'parity_single.jsonl')
        noop = all((a['generated_token_ids_including_sampled_eos'] == b['generated_token_ids_including_sampled_eos'] and a['logprobs'] == b['logprobs'] for a,
            b in zip(clean,
            zero)))
        batch_error = max((abs(a['logprobs'][0][str(k)] - b['logprobs'][0][str(k)]) for a,
            b in zip(clean,
            single) for k in cfg['diagnostic_token_ids']))
        batch_alignment = len(clean) == len(single) and all(
            a['prompt_id'] == b['prompt_id']
            and a['prompt_token_ids'] == b['prompt_token_ids']
            and a['sampling_seed'] == b['sampling_seed']
            and a['generated_token_ids_including_sampled_eos'] == b['generated_token_ids_including_sampled_eos']
            and a['stop_reason'] == b['stop_reason']
            for a, b in zip(clean, single))
        eos = read_jsonl(out / 'parity_eos.jsonl')
        cap = read_jsonl(out / 'parity_cap.jsonl')[0]
        eos_ok=all(row['stop_reason']=='sampled_eos' and row['generated_token_ids_including_sampled_eos']==[row['_forced_eos']] for row in eos)
        cap_ok = cap['stop_reason'] == 'length_cap' and len(cap['generated_token_ids_including_sampled_eos']) == 1 and (cap['generated_token_ids_including_sampled_eos'][-1] != tok.eos_token_id)
        ok = noop and batch_alignment and eos_ok and cap_ok and (batch_error <= cfg['parity_logprob_atol']) and all((v['max_abs_logprob_error'] <= cfg['parity_logprob_atol'] for v in error.values()))
        report = {'status': 'PASS' if ok else 'FAIL',
            'processed_logprobs': True,
            'hf_comparison': error,
            'zero_processor_exact_noop': noop,
            'batch_size_first_context_logprob_max_error': batch_error,
            'batch_size_prompt_seed_trajectory_alignment': batch_alignment,
            'sampled_eos_retained': eos_ok,
            'eos_ids_tested': cfg['_eos_token_ids'],
            'length_cap_no_fictitious_eos': cap_ok,
            'signed_temperature_scaling_verified': all((error[k]['max_abs_logprob_error'] <= cfg['parity_logprob_atol'] for k in ['plus',
            'minus'])),
            'request_specific_weights': [1.0,
            -1.0],
            'tolerance': cfg['parity_logprob_atol'],
            'estimator': 'numerically validated HF approximation to vLLM, not bit-exact across kernels'}
        write_json(out / 'vllm_parity.json', report)
        if not ok:
            raise RuntimeError('vLLM/HF parity failed; see vllm_parity.json')
        return report
    except Exception as exc:
        path = out / 'vllm_parity.json'
        if not path.exists():
            write_json(path, {'status': 'FAIL', 'error': str(exc)})
        raise

def _smoke_child(cfg, mode):
    from .clean_rollouts import generate
    out = output_dir(cfg)
    rows = read_jsonl(out / 'perturb_smoke_contexts.jsonl')
    if mode == 'smoke-clean':
        rows = [r for r in rows if r['eta_label'] == 'zero']
        results = generate(cfg, rows, processor=False, diagnostic=True)
        write_jsonl(out / 'perturb_smoke_clean.jsonl', results)
    elif mode == 'smoke-generate':
        results = generate(cfg,
            rows,
            etas=np.asarray([r['eta'] for r in rows]),
            weights=np.asarray([r['_weights'] for r in rows]),
            processor=True,
            diagnostic=True)
        write_jsonl(out / 'perturb_smoke_rollouts.jsonl', results)
    elif mode == 'smoke-hf':
        import torch
        from .replay_scores import load_model, response_logits
        model = load_model(cfg)
        features = torch.as_tensor(load_npz(out / 'readout_features.npz')['F_vocab'], device='cuda', dtype=torch.float32)
        rows = read_jsonl(out / 'perturb_smoke_rollouts.jsonl')
        errors = []
        for row in rows:
            logits = response_logits(model, row) / cfg['temperature']
            bias = features @ torch.as_tensor(row['_weights'], device='cuda', dtype=torch.float32)
            logits += float(row['eta']) * bias[None, :]
            lp = torch.log_softmax(logits, dim=-1)
            errors.extend((abs(float(lp[t, int(k)]) - v) for t, step in enumerate(row['logprobs']) for k, v in step.items()))
        write_json(out / 'perturb_smoke_hf.json',
            {'max_abs_logprob_error': float(max(errors)),
            'mean_abs_logprob_error': float(np.mean(errors)),
            'comparisons': len(errors)})
    else:
        raise ValueError(mode)

def run_perturb_smoke(cfg):
    from .key_and_statistic import key_coordinates
    cfg = frozen_config(cfg)
    out = output_dir(cfg)
    data = load_npz(out / 'calibration_features.npz')
    bridge = load_npz(out / 'bridge.npz')
    key = load_npz(out / 'keys.npz')['R'][0]
    _, weights, _ = key_coordinates(data['z'][:3], key, bridge)
    lookup = {str(r['prompt_id']): r for r in read_jsonl(out / 'calibration_rollouts.jsonl')}
    features = load_npz(out / 'readout_features.npz')['F_vocab']
    delta = read_json(out / 'delta_calibration.json')['delta']
    diagnostic_ids = set()
    for w in weights:
        bias = features @ w
        diagnostic_ids.update([int(bias.argmin()), int(bias.argmax())])
    rows = []
    for label, eta in [('minus', -delta), ('zero', 0.0), ('plus', delta)]:
        for index, pid in enumerate(data['prompt_id'][:len(weights)]):
            row = dict(lookup[str(pid)])
            row.update(prompt_id=f'{pid}:smoke:{label}',
                source_prompt_id=str(pid),
                eta_label=label,
                eta=float(eta),
                _weights=weights[index].tolist())
            rows.append(row)
    cfg = dict(cfg, max_tokens=min(8, cfg['max_tokens']), diagnostic_token_ids=sorted(diagnostic_ids))
    path = out / 'perturb_smoke_config.json'
    write_json(path, cfg)
    write_jsonl(out / 'perturb_smoke_contexts.jsonl', rows)
    try:
        for mode in ['smoke-clean', 'smoke-generate', 'smoke-hf']:
            subprocess.run([sys.executable,
                '-m',
                'ksrf_exp1.validate_vllm_parity',
                '--config',
                str(path),
                '--worker',
                mode],
                check=True)
        results = read_jsonl(out / 'perturb_smoke_rollouts.jsonl')
        clean = read_jsonl(out / 'perturb_smoke_clean.jsonl')
        by_id = {(r['source_prompt_id'], r['eta_label']): r for r in results}
        zero_ok = all((r['logprobs'] == by_id[r['source_prompt_id'],
            'zero']['logprobs'] and r['generated_token_ids_including_sampled_eos'] == by_id[r['source_prompt_id'],
            'zero']['generated_token_ids_including_sampled_eos'] for r in clean))
        signed = []
        for index, pid in enumerate(data['prompt_id'][:len(weights)]):
            pid = str(pid)
            bias = features @ weights[index]
            low, high = (int(bias.argmin()), int(bias.argmax()))
            base = by_id[pid, 'zero']['logprobs'][0]
            base_odds = base[str(high)] - base[str(low)]
            for label, eta in [('minus', -delta), ('plus', delta)]:
                lp = by_id[pid, label]['logprobs'][0]
                observed = lp[str(high)] - lp[str(low)] - base_odds
                expected = float(eta * (bias[high] - bias[low]))
                valid = observed * expected > 0 and abs(observed - expected) <= max(cfg['parity_logprob_atol'],
                    0.1 * abs(expected))
                signed.append({'prompt_id': pid,
                    'label': label,
                    'observed_logodds_shift': observed,
                    'expected_logodds_shift': expected,
                    'pass': bool(valid)})
        hf = read_json(out / 'perturb_smoke_hf.json')
        stop_ok = all((r['stop_reason'] == 'sampled_eos' and r['generated_token_ids_including_sampled_eos'][-1] in r['eos_token_ids'] or (r['stop_reason'] == 'length_cap' and len(r['generated_token_ids_including_sampled_eos']) == cfg['max_tokens']) for r in results))
        ok = zero_ok and stop_ok and all((s['pass'] for s in signed)) and (hf['max_abs_logprob_error'] <= cfg['parity_logprob_atol'])
        report = {'status': 'PASS' if ok else 'FAIL',
            'zero_noop_exact': zero_ok,
            'signed_logodds': signed,
            'hf_processed_logprob_comparison': hf,
            'stop_metadata_consistent': stop_ok,
            'prompt_count': len(weights),
            'delta': delta,
            'request_specific_bias': True,
            'generation': 'mixed zero/plus/minus batched requests in one vLLM instance; clean no-processor instance isolated'}
        write_json(out / 'perturb_smoke.json', report)
        if not ok:
            raise RuntimeError('Real-readout perturbation smoke failed')
        return report
    except Exception as exc:
        if not (out / 'perturb_smoke.json').exists():
            write_json(out / 'perturb_smoke.json', {'status': 'FAIL', 'error': str(exc)})
        raise
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--worker', required=True)
    a = p.parse_args()
    _child(read_json(a.config), a.worker)
