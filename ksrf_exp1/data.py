"""Frozen Hugging Face revisions and disjoint prompt selection."""
from .common import checksum, output_dir, read_json, write_json, write_jsonl

def frozen_config(cfg):
    path = output_dir(cfg) / 'manifest.json'
    if not path.exists():
        return dict(cfg)
    frozen = read_json(path)['config']
    merged = dict(frozen)
    merged.update(cfg)
    for key in ['model_revision', 'tokenizer_revision', 'encoder_revision', 'dataset_revision']:
        merged[key] = frozen[key]
    return merged

def run(cfg):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from datasets import load_dataset
    api = HfApi()
    cfg = dict(cfg)
    for key, repo_type in [('model', 'model'), ('encoder', 'model'), ('dataset', 'dataset')]:
        cfg[key + '_revision'] = api.repo_info(cfg[key], repo_type=repo_type, revision=cfg[key + '_revision']).sha
    cfg['tokenizer_revision'] = cfg['model_revision']
    tok = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['tokenizer_revision'])
    out = output_dir(cfg)
    skipped = []
    seen = set()
    counts = {}
    selections = {}
    for name, split, n in [('calibration',
        'train_gen',
        cfg['calibration_size']),
        ('heldout',
        'test_gen',
        cfg['heldout_size'])]:
        ds = load_dataset(cfg['dataset'], split=split, revision=cfg['dataset_revision']).shuffle(seed=cfg['selection_seed'])
        records = []
        for row in ds:
            text = row['prompt']
            pid = str(row['prompt_id'])
            raw = tok.encode(text, add_special_tokens=False)
            reason = 'duplicate' if pid in seen or checksum(text) in seen else 'token_length' if not cfg['prompt_min_tokens'] <= len(raw) <= cfg['prompt_max_tokens'] else None
            if reason:
                skipped.append({'prompt_id': pid, 'split': split, 'reason': reason})
                continue
            seen.update([pid, checksum(text)])
            ids = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)
            records.append(dict(prompt_id=pid,
                prompt_text=text,
                prompt_token_ids=ids,
                sampling_seed=cfg['sampling_seed'] + len(records) + (0 if name == 'calibration' else cfg['calibration_size'])))
            if len(records) == n:
                break
        if len(records) != n:
            raise RuntimeError('Insufficient eligible unique prompts: ' + name)
        write_jsonl(out / (name + '_prompts.jsonl'), records)
        counts[name] = len(records)
        selections[name] = checksum([(r['prompt_id'], r['prompt_token_ids']) for r in records])
    write_jsonl(out / 'skipped_prompts.jsonl', skipped)
    write_json(out / 'manifest.json',
        {'config': cfg,
        'counts': counts,
        'selection_checksums': selections,
        'status': 'PREPARED'})
    return cfg
