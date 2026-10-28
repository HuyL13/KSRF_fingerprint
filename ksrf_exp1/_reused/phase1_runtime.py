"""Code extracted from phase1_if_analysis/src/phase1/runtime.py. See docs/LOCAL_CODE_REUSE.md."""
from __future__ import annotations

import numpy as np


def device_of(model):
    return model.get_input_embeddings().weight.device if hasattr(model, 'get_input_embeddings') else next(model.parameters()).device


def prompt_ids(tokenizer, query, config):
    if 'prompt_token_ids' in query:
        ids = query['prompt_token_ids']
    elif config.get('prompt_mode', 'raw') == 'chat':
        messages = query.get('messages')
        if messages is None:
            messages = []
            if config.get('system_prompt'):
                messages.append(dict(role='system', content=config['system_prompt']))
            messages.append(dict(role='user', content=query['prompt']))
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    else:
        ids = tokenizer.encode(query['prompt'], add_special_tokens=config.get('add_special_tokens', True))
    if not ids or len(ids) > config.get('max_prompt_length', 2048):
        raise ValueError('Empty/overlength prompt; inputs are never silently truncated')
    return list(ids)


def teacher_forced_logits(model, tokenizer, query, config):
    import torch
    x = prompt_ids(tokenizer, query, config)
    y = query.get('target_token_ids')
    if y is None:
        y = tokenizer.encode(query['target'], add_special_tokens=False)
    if not y:
        raise ValueError('Target sequence must contain at least one token')
    ids = torch.tensor([x+list(y)], dtype=torch.long, device=device_of(model))
    maximum = getattr(getattr(model, 'config', None), 'max_position_embeddings', None)
    if maximum is not None and ids.shape[1] > maximum:
        raise ValueError('Teacher-forcing sequence exceeds model context')
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    # Causal shift: final prompt position predicts the first target token.
    # KSRF adaptation: keep full-vocabulary logits on the model device.
    # Moving to CPU and back for every trajectory would add a large transfer.
    z = output.logits[0, len(x)-1:len(x)+len(y)-1].float()
    return z, np.asarray(y, dtype=int)
