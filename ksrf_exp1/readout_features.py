"""Calibration-only PCA of actual output-head token rows."""
import numpy as np
from .common import checksum, output_dir, read_jsonl, save_npz, write_json
from .data import frozen_config

def run(cfg):
    import torch
    from transformers import AutoTokenizer
    from .replay_scores import load_model
    cfg = frozen_config(cfg)
    tok = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['tokenizer_revision'])
    special = set(tok.all_special_ids)
    rows = read_jsonl(output_dir(cfg) / 'calibration_rollouts.jsonl')
    ids = np.asarray([v for r in rows for v in r['generated_token_ids_including_sampled_eos'] if v not in special],
        dtype=np.int64)
    if len(ids) < cfg['readout_sample_size']:
        raise ValueError('Insufficient calibration response tokens for PCA')
    rng = np.random.default_rng(cfg['projection_seed'])
    chosen = ids[rng.choice(len(ids), cfg['readout_sample_size'], replace=False)]
    model = load_model(cfg)
    W = model.get_output_embeddings().weight.detach()
    sample = W[torch.as_tensor(chosen, device=W.device)].float()
    mean = sample.mean(0)
    centered = sample - mean
    cov = centered.T @ centered / (len(sample) - 1)
    _, vectors = torch.linalg.eigh(cov)
    P = vectors[:, -cfg['readout_rank']:].flip(1).contiguous()
    features = torch.empty((W.shape[0], cfg['readout_rank']), dtype=torch.float32, device='cpu')
    for start in range(0, len(W), 2048):
        features[start:start + 2048] = (W[start:start + 2048].float() @ P).cpu()
    save_npz(output_dir(cfg) / 'readout_features.npz',
        P=P.cpu().numpy(),
        mean=mean.cpu().numpy(),
        F_vocab=features.numpy(),
        sample_token_ids=chosen)
    write_json(output_dir(cfg) / 'readout_metadata.json',
        {'fit_split': 'calibration',
        'sample_size': len(chosen),
        'sample_checksum': checksum(chosen.tolist()),
        'features': 'actual_lm_head @ P, uncentered',
        'orthogonality_max_error': float((P.T @ P - torch.eye(P.shape[1],
        device=P.device)).abs().max())})
    del model, W, sample, cov, P, features
    torch.cuda.empty_cache()
