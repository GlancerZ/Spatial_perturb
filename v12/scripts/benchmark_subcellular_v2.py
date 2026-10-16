"""Find a conservative full-backbone training batch on one allocated GPU."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import torch
import yaml
from spatial_pretrain.models.subcellular_v2 import SubcellularV2
from train_subcellular_v2 import losses


def synthetic_batch(model, cfg, size):
    cells = cfg['data']['context_cells']
    n = cfg['data']['context_tokens_per_compartment']
    length = 1+2*n
    query = 2*cfg['data']['query_genes']
    candidates = torch.tensor([v for k, v in model.vocab.items() if not k.startswith('<')], device='cuda')
    genes = candidates[torch.randint(len(candidates), (size, cells, length), device='cuda')]
    genes[..., 0] = model.vocab['<cls>']
    comp = torch.cat([torch.full((1,), 2), torch.zeros(n), torch.ones(n)]).long().cuda()
    comp = comp.expand(size, cells, length).clone()
    values = torch.randint(1, 51, genes.shape, device='cuda').float(); values[..., 0] = -2
    query_comp = (torch.arange(query, device='cuda') >= query//2).long().expand(size, query)
    return dict(gene_ids=genes, input_bins=values, compartment_ids=comp,
        token_valid_mask=torch.ones_like(genes, dtype=torch.bool),
        node_indices=torch.arange(cells, device='cuda').expand(size, cells),
        coords=torch.randn(size, cells, 2, device='cuda')*40,
        query_gene_ids=candidates[torch.randint(len(candidates), (size, query), device='cuda')],
        query_compartment_ids=query_comp, query_valid_mask=torch.ones(size, query, dtype=torch.bool, device='cuda'),
        target_log_counts=torch.rand(size, query, device='cuda')*3,
        target_nonzero=torch.rand(size, query, device='cuda') > .5,
        neighbor_query_mask=query_comp.eq(1), task_mode=torch.zeros(size, dtype=torch.long, device='cuda'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    model = SubcellularV2(cfg['assets_dir'], **cfg['model']).cuda().train()
    model.set_backbone_trainable(full=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    results = []; selected = None
    total_memory = torch.cuda.get_device_properties(0).total_memory
    for size in [1, 2, 4, 8, 16]:
        if size > args.max_batch: break
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        try:
            batch = synthetic_batch(model, cfg, size)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(batch)
                    loss, parts = losses(output, batch, cfg, 100)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite capacity-test loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                del output, loss
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            row = dict(batch=size, ok=True, full_tokens_per_cell=1+2*cfg['data']['context_tokens_per_compartment'],
                seconds_per_update=(time.monotonic()-began)/2,
                allocated_gib=allocated/2**30, reserved_gib=reserved/2**30,
                fraction=allocated/total_memory)
            if allocated <= .80*total_memory: selected = size
            results.append(row); print('[capacity]', json.dumps(row), flush=True)
            del batch
            if allocated > .80*total_memory: break
        except torch.OutOfMemoryError as error:
            results.append(dict(batch=size, ok=False, error='CUDA out of memory'))
            print(f'[capacity-oom] batch={size}', flush=True)
            break
    optimizer.zero_grad(set_to_none=True)
    if selected is None: raise RuntimeError('No batch passed the 80% memory headroom gate')
    record = dict(selected_batch_size=selected, device=torch.cuda.get_device_name(0),
        total_gib=total_memory/2**30, results=results,
        note='Synthetic maximum-length fully unfrozen training; not a real-data ETA')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    if args.apply:
        run = Path(cfg['output_dir'])
        if (run/'resume.pt').exists() or (run/'checkpoints/step_00000000.pt').exists():
            raise RuntimeError('Cannot change batch after a run has initialized')
        cfg['training']['batch_size'] = selected
        args.config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('[capacity-selected]', selected, flush=True)


if __name__ == '__main__': main()
