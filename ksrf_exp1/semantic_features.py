"""Frozen BGE CLS embeddings and calibration-only affine projections."""
import numpy as np
from .common import load_npz, output_dir, read_jsonl, save_npz, write_json
from .data import frozen_config

def encode_texts(cfg, texts):
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = frozen_config(cfg)
    model = SentenceTransformer(cfg['encoder'], revision=cfg['encoder_revision'], device='cuda', trust_remote_code=False)
    model.eval()
    model.max_seq_length = cfg['encoder_max_length']
    lengths = [len(model.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    trunc = [{'index': i,
        'tokens': n,
        'limit': cfg['encoder_max_length']} for i,
        n in enumerate(lengths) if n > cfg['encoder_max_length']]
    embedding = model.encode(texts,
        batch_size=cfg['embedding_batch_size'],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        prompt='')
    norms = np.linalg.norm(embedding, axis=1)
    if not np.isfinite(embedding).all() or (norms <= 0).any():
        raise ValueError('Invalid BGE embedding')
    del model
    torch.cuda.empty_cache()
    return (embedding.astype(np.float32), trunc)

def transform_embeddings(cfg, prompt, response):
    t = load_npz(output_dir(cfg) / 'semantic_transform.npz')
    phi = (response - t['response_mean']) @ t['response_basis']
    z = (prompt - t['prompt_mean']) @ t['prompt_basis'] @ t['prompt_whitening']
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    if not np.isfinite(z).all() or (norm <= 0).any():
        raise ValueError('Invalid prompt projection')
    return (phi, z / norm)

def run(cfg):
    from sklearn.decomposition import PCA
    cfg = frozen_config(cfg)
    out = output_dir(cfg)
    all_data = {}
    truncation = {}
    for split in ['calibration', 'heldout']:
        data = load_npz(out / (split + '_features.npz'))
        by_id = {r['prompt_id']: r for r in read_jsonl(out / (split + '_rollouts.jsonl'))}
        rows = [by_id[str(i)] for i in data['prompt_id']]
        texts = [r['prompt_text'] for r in rows] + [r['response_text_without_special_tokens'] for r in rows]
        embedding, trunc = encode_texts(cfg, texts)
        n = len(rows)
        data.update(prompt_embedding=embedding[:n], response_embedding=embedding[n:])
        all_data[split] = data
        truncation[split] = trunc
    cal = all_data['calibration']
    response = PCA(n_components=cfg['response_rank'], svd_solver='full').fit(cal['response_embedding'])
    prompt = PCA(n_components=cfg['key_rank'], svd_solver='full').fit(cal['prompt_embedding'])
    variance = prompt.explained_variance_.astype(np.float64)
    reg = cfg['whitening_regularization'] * variance.mean()
    if reg <= 0:
        raise ValueError('Degenerate prompt covariance')
    save_npz(out / 'semantic_transform.npz',
        response_mean=response.mean_,
        response_basis=response.components_.T,
        prompt_mean=prompt.mean_,
        prompt_basis=prompt.components_.T,
        prompt_whitening=np.diag(1 / np.sqrt(variance + reg)),
        prompt_variance=variance,
        whitening_regularization=np.asarray(reg),
        encoder_revision=np.asarray(cfg['encoder_revision']))
    for split, data in all_data.items():
        data['phi'], data['z'] = transform_embeddings(cfg, data['prompt_embedding'], data['response_embedding'])
        save_npz(out / (split + '_features.npz'), **data)
    write_json(out / 'semantic_metadata.json',
        {'encoder': cfg['encoder'],
        'encoder_revision': cfg['encoder_revision'],
        'pooling': 'checkpoint SentenceTransformer modules (BGE CLS)',
        'normalization': 'unit L2 before PCA',
        'prompt_instruction': None,
        'fit_split': 'calibration',
        'truncation': truncation})
