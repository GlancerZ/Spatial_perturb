"""Compute-only V5 acceptance on real cache: anchor identity and two-step gradients."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from spatial_pretrain.data.subcellular_v5 import V5Dataset, collate_v5, encode_whole_cell_v5
from spatial_pretrain.models.subcellular_v5 import SubcellularV5
from spatial_pretrain.training.subcellular_v4 import losses
from train_subcellular_v5 import validate_config, atomic_json
from qa_subcellular_v4 import critical_parameters
from evaluate_plain_scgpt_baseline import load_ordinary_collator


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID') or not torch.cuda.is_available():
        raise RuntimeError('V5 QA requires a verified Slurm GPU compute step')
    if args.output.exists() or args.batch_size < 1:
        raise ValueError('Use a new report path and a positive batch size')
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(42)
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='subcellular_v5_real_qa_v1', status='running',
                  jobid=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
                  python=sys.executable, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(), started_at=time.time())
    atomic_json(report, args.output)
    try:
        dataset = V5Dataset(cfg, split='train')
        # Uniform cache positions exercise different slides and all three mask
        # modes. Every item performs checked integer recovery on actual counts.
        indices = np.unique(np.linspace(0, len(dataset) - 1, min(64, len(dataset)), dtype=np.int64))
        candidates = [(int(index), dataset[int(index)]) for index in indices]
        modes = {int(item['task_mode']) for _, item in candidates}
        if modes != {0, 1, 2}:
            raise AssertionError(f'Real-cache QA did not cover all task modes: {modes}')
        candidates.sort(key=lambda pair: int(pair[1]['token_valid_mask'].sum()), reverse=True)
        items = [item for _, item in candidates[:args.batch_size]]
        if len(items) != args.batch_size:
            raise ValueError('Insufficient distinct real-cache examples for the requested batch')
        batch = {key: value.cuda() for key, value in collate_v5(items, pad_token_id=dataset.pad_id).items()}
        model = SubcellularV5(cfg['assets_dir'], **cfg['model']).cuda().train()
        # Compare against the actual verified upstream collator, not another
        # transcription of our implementation. No truncation/RNG ambiguity.
        collator, collator_source = load_ordinary_collator(Path(cfg['assets_dir']), model.vocab, model.backbone.config)
        raw_n = np.array([0, 1, 2, 2, 7, 5, 0, 3], dtype=np.int64)
        raw_c = np.array([2, 1, 0, 2, 1, 0, 0, 3], dtype=np.int64)
        panel_ids = np.array([identifier for token, identifier in model.vocab.items()
                              if token not in ('<pad>', '<cls>', '<eoc>')][:8], dtype=np.int64)
        ours = encode_whole_cell_v5(raw_n, raw_c, panel_ids, pad_token_id=model.pad_id,
                                    cls_token_id=model.cls_id, rng=np.random.RandomState(73))
        total = raw_n + raw_c
        nonzero = total > 0
        numpy_state = np.random.get_state()
        try:
            np.random.seed(73)
            official = collator([dict(id=0, genes=torch.tensor(np.r_[model.cls_id, panel_ids[nonzero]]),
                expressions=torch.tensor(np.r_[model.pad_value, total[nonzero]], dtype=torch.float32))])
        finally:
            np.random.set_state(numpy_state)
        np.testing.assert_array_equal(ours['whole_gene_ids'], official['gene'][0].numpy())
        np.testing.assert_array_equal(ours['whole_input_bins'], official['expr'][0].numpy())
        before_hash = state_digest(model.backbone)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            initial = model(batch, return_embeddings=True)
            genes, bins, valid = (batch[key] for key in
                                  ('whole_gene_ids', 'whole_input_bins', 'whole_token_valid_mask'))
            reference = model.backbone(genes, bins, ~valid,
                torch.zeros((*genes.shape, model.backbone.dim), device='cuda'))
            reference = F.normalize(reference[:, 0].float(), dim=-1, eps=1e-8)
            for key in ('base', 'joint', 'intrinsic'):
                torch.testing.assert_close(initial['embeddings'][key], reference, atol=0, rtol=0)
            changed = dict(batch, target_bins=(batch['target_bins'] + 7).remainder(51))
            altered = model(changed, return_embeddings=True)
            torch.testing.assert_close(altered['embeddings']['joint'], reference, atol=0, rtol=0)
        del initial, altered, reference
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=cfg['training']['lr_new'],
                                     weight_decay=cfg['training']['weight_decay'])
        critical = critical_parameters(model)
        critical['residual_projection.weight'] = model.residual_projection.weight
        if not cfg['model']['spatial_enabled']:
            critical = {name: value for name, value in critical.items() if not name.startswith('spatial_blocks.')}
        stages = []
        torch.cuda.reset_peak_memory_stats()
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch)
                loss, parts = losses(output, batch, cfg, step)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite V5 real-cache loss')
            loss.backward()
            gradients = {name: 0. if p.grad is None else float(p.grad.detach().float().norm())
                         for name, p in critical.items()}
            if gradients['residual_projection.weight'] <= 0:
                raise AssertionError('Zero-initialized residual projection cannot learn')
            if step == 2 and any(not np.isfinite(value) or value <= 0 for value in gradients.values()):
                raise AssertionError(f'V5 required path has no finite gradient after projection opens: {gradients}')
            if any(p.grad is not None for p in model.backbone.parameters()):
                raise AssertionError('Frozen scGPT received gradients')
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['training']['grad_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            stages.append(dict(step=step, loss=parts, gradient_norm=float(norm), required_gradients=gradients))
        if before_hash != state_digest(model.backbone):
            raise AssertionError('The frozen plain-scGPT reference changed during adaptation')
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('V5 optimizer produced nonfinite parameters')
        report.update(status='passed', batch_size=len(items), cache_items_checked=len(candidates),
                      mask_modes=sorted(modes), step_zero_identity_verified=True,
                      unchanged_backbone_verified=True, target_field_isolation_verified=True,
                      official_binning_same_rng_verified=True, collator_provenance=collator_source,
                      backbone_provenance=model.backbone.provenance, arch_config=model.arch_config,
                      stages=stages, max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      limitation='Two QA updates only; no training checkpoint is written; not evidence of efficacy')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_at'] = time.time()
        atomic_json(report, args.output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
