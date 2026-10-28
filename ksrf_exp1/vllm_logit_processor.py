"""vLLM request adapter; upstream owns request movement and removal."""
import numpy as np
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
try:
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as V2LogitsProcessor
except ImportError:
    class V2LogitsProcessor:
        """Older installations expose only the V1 runner interface."""

class KSRFLogitsProcessor(AdapterLogitsProcessor, V2LogitsProcessor):

    @classmethod
    def validate_params(cls, sampling_params):
        extra = sampling_params.extra_args or {}
        eta = float(extra.get('ksrf_eta', 0.0))
        if not np.isfinite(eta):
            raise ValueError('KSRF eta must be finite')
        if eta == 0.0:
            return
        temperature = float(sampling_params.temperature)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('KSRF perturbations require positive finite temperature')
        weights = np.asarray(extra.get('ksrf_weights', []), dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all():
            raise ValueError('KSRF requires a finite nonempty weight vector')
        if not isinstance(extra.get('ksrf_features'), str) or not extra['ksrf_features']:
            raise ValueError('KSRF requires a shared readout-feature NPZ path')

    def __init__(self, vllm_config, device, is_pin_memory=None):
        if hasattr(device, 'max_num_reqs'):
            self.device = device.device
        else:
            AdapterLogitsProcessor.__init__(self, vllm_config, device, is_pin_memory)
            self.device = device
        self.features = {}
        self.slot_processors = {}

    def add_request(self, req_idx, sampling_params):
        processor = self.new_req_logits_processor(sampling_params)
        self.slot_processors.pop(req_idx, None)
        if processor is not None:
            self.slot_processors[req_idx] = processor
        return processor is not None

    def apply(self, logits, ctx=None):
        if ctx is None:
            return AdapterLogitsProcessor.apply(self, logits)
        # V2 rows are reordered independently of persistent request slots.
        for req_idx in ctx.idx_mapping_np:
            processor = self.slot_processors.get(int(req_idx))
            if processor is not None:
                mask = ctx.expanded_idx_mapping == int(req_idx)
                logits[mask] = processor([], logits[mask])
        return logits

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        self.validate_params(params)
        extra = params.extra_args or {}
        eta = float(extra.get('ksrf_eta', 0.0))
        if eta == 0.0:
            return None
        path = extra['ksrf_features']
        if path not in self.features:
            with np.load(path, allow_pickle=False) as f:
                self.features[path] = torch.as_tensor(f['F_vocab'], device=self.device, dtype=torch.float32)
        w = torch.as_tensor(extra['ksrf_weights'], device=self.device, dtype=torch.float32)
        if w.numel() != self.features[path].shape[1]:
            raise ValueError('Readout rank and request weight dimensions differ')
        bias = self.features[path] @ w * (float(params.temperature) * eta)

        def apply(output_ids, logits):
            if logits.shape[-1] != bias.shape[0]:
                raise RuntimeError('Vocabulary mismatch')
            logits.add_(bias)
            return logits
        return apply
