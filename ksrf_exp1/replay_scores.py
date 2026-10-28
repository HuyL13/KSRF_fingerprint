"""Teacher-forced clean trajectory scores at exact sampled token IDs."""
import numpy as np
from .common import checksum, load_npz, output_dir, read_jsonl, save_npz, write_json
from .data import frozen_config
from ._reused.phase1_runtime import teacher_forced_logits

def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    cfg = frozen_config(cfg)
    return AutoModelForCausalLM.from_pretrained(cfg['model'],
        revision=cfg['model_revision'],
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        attn_implementation='sdpa').eval().requires_grad_(False)

def response_logits(model, record):
    prompt = record['prompt_token_ids']
    sampled = record['generated_token_ids_including_sampled_eos']
    if not prompt or not sampled:
        raise ValueError('Empty trajectory')
    logits, _ = teacher_forced_logits(
        model, None, {'prompt_token_ids': prompt, 'target_token_ids': sampled},
        {'max_prompt_length': len(prompt)})
    return logits

def score_logits(logits, sampled, features, temperature, chunk_size=16):
    import torch
    total = torch.zeros(features.shape[1], device=features.device, dtype=torch.float32)
    for start in range(0, len(sampled), chunk_size):
        p = torch.softmax(logits[start:start + chunk_size].float() / temperature, dim=-1)
        ids = torch.tensor(sampled[start:start + chunk_size], device=features.device)
        total += (features[ids] - p @ features).sum(0)
    return total

def run(cfg):
    import torch
    cfg = frozen_config(cfg)
    model = load_model(cfg)
    f = torch.as_tensor(load_npz(output_dir(cfg) / 'readout_features.npz')['F_vocab'], device='cuda', dtype=torch.float32)
    for split in ['calibration', 'heldout']:
        rows = read_jsonl(output_dir(cfg) / (split + '_rollouts.jsonl'))
        gamma = []
        kept = []
        excluded = []
        for r in rows:
            if not r['response_text_without_special_tokens'].strip():
                excluded.append(r['prompt_id'])
                continue
            logits = response_logits(model, r)
            gamma.append(score_logits(logits,
                r['generated_token_ids_including_sampled_eos'],
                f,
                cfg['temperature'],
                cfg['replay_chunk_size']).cpu().numpy())
            kept.append(r)
        save_npz(output_dir(cfg) / (split + '_features.npz'),
            Gamma=np.asarray(gamma,
            dtype=np.float32),
            prompt_id=np.asarray([r['prompt_id'] for r in kept]),
            sequence_length=np.asarray([len(r['generated_token_ids_including_sampled_eos']) for r in kept]),
            stop_reason=np.asarray([r['stop_reason'] for r in kept]))
        write_json(output_dir(cfg) / (split + '_replay_metadata.json'),
            {'count': len(kept),
            'excluded_empty_response': excluded,
            'trajectory_checksum': checksum([(r['prompt_id'],
            r['prompt_token_ids'],
            r['generated_token_ids_including_sampled_eos']) for r in kept]),
            'distribution': 'softmax(HF_raw_logits / temperature)',
            'exactness_requires': 'vllm_parity.json PASS'})
    del model, f
    torch.cuda.empty_cache()
