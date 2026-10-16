"""Two-GPU capacity probe with all 512 token slots valid in every cell."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from spatial_pretrain.config import load_config
from spatial_pretrain.models.compartment import CompartmentSSGTModel
from spatial_pretrain.training.runner import _build_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--config', default='configs/xenium_compartment_full.yaml')
    parser.add_argument('--output-prefix', default='batch_probe')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    torch.manual_seed(42)
    cfg = load_config(ROOT / args.config)
    artifact = torch.load(cfg.model.genept_path, map_location='cpu')
    embeddings = artifact['embeddings'].float()
    genes = len(embeddings)
    embeddings = torch.cat((embeddings, torch.zeros_like(embeddings[:1])))
    from spatial_pretrain.models.spatial_bias import spatial_kwargs
    raw = CompartmentSSGTModel(num_genes=genes+1, num_value_bins=32, hidden_dim=256,
        num_layers=6, gene_heads=16, cell_heads=cfg.model.cell_heads, dropout=0.1, ffn_mult=4,
        gene_padding_idx=genes, gene_embedding_init=embeddings, gradient_checkpointing=True,
        cell_attention_mode=cfg.model.cell_attention_mode, nuclear_tokens=256, cytoplasmic_tokens=256,
        cell_token_dim=16, require_flash_attention=True, **spatial_kwargs(cfg.model)).cuda()
    model = DDP(raw, device_ids=[rank])
    optimizer, stats = _build_optimizer(model.module, cfg)
    shape = (args.batch, 256, 512)
    comp = torch.zeros(shape, dtype=torch.long, device='cuda')
    comp[:, :, 256:] = 1
    target = torch.randint(1, 32, shape, device='cuda')
    mask = torch.rand(shape, device='cuda') < 0.6
    inputs = dict(gene_ids=torch.randint(0, genes, shape, device='cuda'),
        input_bins=target.masked_fill(mask, 32), compartment_ids=comp,
        cell_roles=torch.zeros(args.batch, 256, dtype=torch.long, device='cuda'),
        node_indices=torch.arange(256, device='cuda').expand(args.batch, -1),
        token_valid_mask=torch.ones(shape, dtype=torch.bool, device='cuda'))
    inputs['coords'] = torch.stack((torch.arange(256, device='cuda') % 16,
                                   torch.arange(256, device='cuda') // 16), -1).float()[None].expand(args.batch, -1, -1) * 20
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(**inputs)['mask_logits']
            loss = torch.nn.functional.cross_entropy(logits[mask], target[mask])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError('Non-finite loss or gradient in batch probe')
        optimizer.step()
        torch.cuda.synchronize()
        if rank == 0:
            print(f'[batch-probe] batch_per_gpu={args.batch} update={step+1} loss={loss.item():.5f}', flush=True)
    memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device='cuda')
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    elapsed = time.monotonic() - start
    if rank == 0:
        result = dict(batch_per_gpu=args.batch, peak_allocated_gib=memory[0].item()/2**30,
            peak_reserved_gib=memory[1].item()/2**30, total_gib=torch.cuda.get_device_properties(rank).total_memory/2**30,
            seconds_per_update=elapsed/3, valid_tokens_per_cell=512, cells_per_subgraph=256,
            precision='bfloat16', flash_attention=('within_cell_required_cross_cell_auto_sdpa'
                if cfg.model.cell_spatial_bias else 'required_no_fallback'), optimizer=cfg.optimizer.name, **stats)
        path = ROOT / 'logs' / f'{args.output_prefix}_{args.batch}.json'
        path.write_text(json.dumps(result, indent=2))
        print('[batch-result] '+json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
