"""Actual final-normalization JVP/VJP damped inverse, never dense Jacobians."""
import numpy as np
from .common import load_npz, output_dir, read_jsonl, write_json
from .data import frozen_config

def damped_tangent(norm, state, target, damping, maxiter):
    import torch
    from scipy.sparse.linalg import LinearOperator, cg
    r = state.detach().float()
    target = target.detach().float()

    def fn(x):
        return norm(x)
    _, pullback = torch.func.vjp(fn, r)

    def jt(v):
        return pullback(v)[0]

    def j(v):
        return torch.func.jvp(fn, (r,), (v,))[1]

    def matvec(v):
        x = torch.as_tensor(v, device=r.device, dtype=r.dtype)
        return (jt(j(x)) + damping * x).detach().cpu().numpy().astype(np.float64)
    rhs = jt(target).detach().cpu().numpy().astype(np.float64)
    operator = LinearOperator((r.numel(), r.numel()), matvec=matvec, dtype=np.float64)
    shift, info = cg(operator, rhs, rtol=1e-05, atol=0.0, maxiter=maxiter)
    delta = torch.as_tensor(shift, device=r.device, dtype=r.dtype)
    achieved = j(delta)
    return {'relative_tangent_residual': float(((target - achieved).norm() / target.norm()).detach()),
        'required_residual_shift_norm': float(delta.norm() / r.norm()),
        'normal_equation_relative_residual': float(np.linalg.norm(matvec(shift) - rhs) / max(np.linalg.norm(rhs),
        1e-30)),
        'solver_info': int(info),
        'damping': float(damping)}

def run(cfg):
    import torch
    from .replay_scores import load_model
    cfg = frozen_config(cfg)
    out = output_dir(cfg)
    model = load_model(cfg)
    norm = model.model.norm
    P = torch.as_tensor(load_npz(out / 'readout_features.npz')['P'], device='cuda', dtype=torch.float32)
    bridge = load_npz(out / 'bridge.npz')
    keys = load_npz(out / 'keys.npz')['R']
    data = load_npz(out / 'heldout_features.npz')
    by_id = {r['prompt_id']: r for r in read_jsonl(out / 'heldout_rollouts.jsonl')}
    diagnostics = []
    saved = []

    def capture(module, args):
        saved.append(args[0].detach())
    handle = norm.register_forward_pre_hook(capture)
    try:
        for index, pid in enumerate(data['prompt_id'][:cfg['tangent_prompts']]):
            record = by_id[str(pid)]
            tokens = record['prompt_token_ids'] + record['generated_token_ids_including_sampled_eos']
            saved.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([tokens], device='cuda'), use_cache=False)
            states = saved[-1][0]
            length = len(record['prompt_token_ids'])
            T = len(record['generated_token_ids_including_sampled_eos'])
            w = bridge['L'] @ bridge['V8'] @ (keys[0] @ data['z'][index])
            target = P @ torch.as_tensor(w, device='cuda', dtype=torch.float32)
            if target.norm() <= 0:
                raise ValueError('Zero tangent target')
            for pos in np.linspace(length - 1, length + T - 2, min(T, cfg['tangent_states_per_prompt']), dtype=int):
                result = damped_tangent(norm, states[pos], target, cfg['tangent_damping'], cfg['tangent_maxiter'])
                diagnostics.append(dict(prompt_id=str(pid), prediction_position=int(pos), **result))
    finally:
        handle.remove()
    residual = np.asarray([d['relative_tangent_residual'] for d in diagnostics])
    shift = np.asarray([d['required_residual_shift_norm'] for d in diagnostics])
    if not len(residual):
        raise ValueError('No tangent states')
    report = {'status': 'PASS' if all((d['solver_info'] == 0 for d in diagnostics)) and np.percentile(residual,
        95) <= cfg['tangent_residual_warning'] else 'WARN',
        'normalization_module': type(norm).__qualname__,
        'normalization_epsilon': float(norm.variance_epsilon),
        'jacobian_method': 'torch.func.jvp/vjp of actual frozen final norm in FP32',
        'solver': 'scipy CG on J^T J + damping I',
        'states': len(diagnostics),
        'relative_tangent_residual': {'median': float(np.median(residual)),
        'p95': float(np.percentile(residual,
        95))},
        'required_residual_shift_norm': {'median': float(np.median(shift)),
        'p95': float(np.percentile(shift,
        95))},
        'details': diagnostics}
    write_json(out / 'tangent_diagnostics.json', report)
    return report
